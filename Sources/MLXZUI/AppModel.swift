import Foundation
import Observation
import Logging
import MLXZCore
import MLXZHub
import MLXZServer

/// The composition-root, main-actor state for the GUI. Holds the shared `ModelManager`,
/// `InferenceServer`, and `LogStore`, and exposes observable properties the SwiftUI views bind to.
///
/// The concrete MLX loader is injected (so this module stays free of MLX and builds under
/// `swift build`); the App target supplies `MLXModelLoader`.
/// Observable holder for live server metrics. Separate from `AppModel` so the server's metrics
/// sink can be wired during `AppModel.init` without capturing the not-yet-initialized `self`.
@MainActor
@Observable
public final class ServerMetrics {
    public private(set) var requestsServed = 0
    public private(set) var lastTokensPerSecond: Double?

    public init() {}

    func record(_ usage: TokenUsage) {
        requestsServed += 1
        if let tps = usage.tokensPerSecond { lastTokensPerSecond = tps }
    }
}

@MainActor
@Observable
public final class AppModel {
    // Server config (bound to the UI), persisted to UserDefaults so an auto-started server uses the
    // user's actual binding across launches.
    public var host: String = "127.0.0.1" {
        didSet { defaults.set(host, forKey: Keys.host) }
    }
    public var port: Int = 8080 {
        didSet { defaults.set(port, forKey: Keys.port) }
    }
    public var bindLAN: Bool = false {
        didSet { defaults.set(bindLAN, forKey: Keys.bindLAN) }
    }
    public var apiKey: String = "" {
        didSet { defaults.set(apiKey, forKey: Keys.apiKey) }
    }
    /// Start the server automatically when the app launches (once the binding is restored). Persisted.
    public var autoStartServer: Bool = false {
        didSet { defaults.set(autoStartServer, forKey: Keys.autoStartServer) }
    }

    // Observable state.
    public private(set) var modelState: ModelManager.State = .empty
    public private(set) var serverRunning: Bool = false
    /// The loaded engine's active speculative-decoding mode ("DSpark drafter: …",
    /// "native MTP"), nil when plain. Drives the ⚡ badge in the model library.
    public private(set) var speculationStatus: String? = nil

    // Live server metrics (an observable holder so the server's sink can update it without
    // capturing `self` during init).
    public let metrics = ServerMetrics()
    public var requestsServed: Int { metrics.requestsServed }
    public var lastTokensPerSecond: Double? { metrics.lastTokensPerSecond }

    public let logStore = LogStore()
    public let catalog = HubCatalog()
    public let localStore = LocalModelStore()
    public let downloads: DownloadManager

    /// When true (default), the model is auto-unloaded on critical memory pressure.
    public var autoUnloadOnMemoryPressure: Bool = true

    private let manager: ModelManager
    private let server: InferenceServer
    private let logger = Logger(label: "mlxz.app")
    private var memoryMonitor: MemoryPressureMonitor?
    private let defaults: UserDefaults

    private enum Keys {
        static let host = "server.host"
        static let port = "server.port"
        static let bindLAN = "server.bindLAN"
        static let apiKey = "server.apiKey"
        static let autoStartServer = "server.autoStart"
    }

    /// User-facing performance settings (KV bits, prefix-cache slots, snapshot block), persisted and
    /// applied on the next model load via the loader's perf provider.
    public let perfSettings: PerfSettings

    public init(
        loader: any ModelLoading,
        downloader: any ModelDownloading,
        embeddingLoader: any EmbeddingLoading,
        perfSettings: PerfSettings = PerfSettings(),
        defaults: UserDefaults = .standard
    ) {
        self.defaults = defaults
        // Restore persisted server config (didSet doesn't fire during init, so assign directly).
        if let h = defaults.string(forKey: Keys.host) { self.host = h }
        if defaults.object(forKey: Keys.port) != nil { self.port = defaults.integer(forKey: Keys.port) }
        self.bindLAN = defaults.bool(forKey: Keys.bindLAN)
        if let k = defaults.string(forKey: Keys.apiKey) { self.apiKey = k }
        self.autoStartServer = defaults.bool(forKey: Keys.autoStartServer)

        self.perfSettings = perfSettings
        self.downloads = DownloadManager(downloader: downloader)
        let logStore = self.logStore
        let manager = ModelManager(loader: loader)
        self.manager = manager
        let store = self.localStore
        self.server = InferenceServer(
            manager: manager,
            logSink: { line in
                Task { @MainActor in logStore.append(line) }
            },
            extraModelIDs: { store.installedModels().map(\.descriptor.repoID) },
            embeddingManager: EmbeddingManager(loader: embeddingLoader),
            metricsSink: { [metrics] usage in
                Task { @MainActor in metrics.record(usage) }
            }
        )

        // Auto-unload the model on critical memory pressure to avoid an OS kill.
        memoryMonitor = MemoryPressureMonitor { [weak self] isCritical in
            guard isCritical else { return }
            Task { @MainActor in
                guard let self, self.autoUnloadOnMemoryPressure,
                      self.modelState.loadedDescriptor != nil else { return }
                self.logStore.append("⚠️ Critical memory pressure — unloading model.")
                await self.unload()
            }
        }
    }

    /// Begin observing the manager's state stream. Call from a SwiftUI `.task {}`.
    public func observeModelState() async {
        for await state in await manager.states() {
            self.modelState = state
        }
    }

    // MARK: - Installed models

    public func installedModels() -> [InstalledModel] {
        localStore.installedModels()
    }

    /// Delete an installed model from the local cache. If it's the currently-loaded model, unload it
    /// first so we don't yank files out from under the running engine.
    public func deleteModel(_ model: InstalledModel) async {
        if modelState.loadedDescriptor?.repoID == model.descriptor.repoID {
            await unload()
        }
        if localStore.delete(model) {
            // Drop any lingering download entry so the search row stops offering "Retry" for a model
            // the user has just removed from disk.
            downloads.clear(model.descriptor.repoID)
            logStore.append("Deleted \(model.descriptor.repoID)")
        } else {
            logStore.append("Failed to delete \(model.descriptor.repoID)")
        }
    }

    // MARK: - Downloads

    /// Remove stale terminal download entries (failed/cancelled/done) whose files are no longer on
    /// disk — e.g. a partial download the user deleted — so search rows stop offering "Retry" for them.
    public func pruneStaleDownloads() {
        downloads.pruneStale { [localStore] repoID in
            localStore.hasCacheDirectory(forRepoID: repoID)
        }
    }

    public func startDownload(_ repoID: String) {
        logStore.append("Downloading \(repoID)…")
        downloads.start(repoID)
    }

    public func cancelDownload(_ repoID: String) {
        downloads.cancel(repoID)
    }

    // MARK: - Model lifecycle

    public func load(_ descriptor: ModelDescriptor) async {
        // Auto-attach a matching installed MTP drafter (self-speculative decoding) if present —
        // unless the user disabled the drafter in Performance settings.
        let drafterID = perfSettings.useMTPDrafter
            ? matchingInstalledDrafter(for: descriptor.repoID)
            : nil
        if let drafterID {
            logStore.append("Loading \(descriptor.repoID) + MTP drafter \(drafterID)…")
        } else if let dsparkDrafter = DrafterPairing.dsparkDrafterRepoID(forTarget: descriptor.repoID) {
            logStore.append(
                "Loading \(descriptor.repoID) — DSpark drafter \(dsparkDrafter) auto-attaches (downloads on first use)…")
        } else {
            logStore.append("Loading \(descriptor.repoID)…")
        }
        do {
            try await manager.load(descriptor, draftModelID: drafterID)
            let speculation = await manager.currentEngine()?.speculationStatus
            speculationStatus = speculation
            if let speculation {
                logStore.append("Loaded \(descriptor.repoID) — speculative decoding ON (\(speculation))")
            } else {
                logStore.append("Loaded \(descriptor.repoID)")
            }
        } catch {
            speculationStatus = nil
            logStore.append("Load failed: \(error)")
        }
    }

    /// The repo id of an installed MTP drafter that pairs with `baseRepoID`, if one is present.
    public func matchingInstalledDrafter(for baseRepoID: String) -> String? {
        guard !DrafterPairing.isDrafter(baseRepoID) else { return nil }
        let expected = DrafterPairing.drafterRepoID(forBase: baseRepoID)
        return installedModels().first { $0.descriptor.repoID == expected }?.descriptor.repoID
    }

    public func unload() async {
        await manager.unload()
        speculationStatus = nil
        logStore.append("Unloaded model")
    }

    // MARK: - Server lifecycle

    public func startServer() async {
        let effectiveHost = bindLAN ? "0.0.0.0" : host
        let config = ServerConfig(
            host: effectiveHost,
            port: port,
            apiKey: apiKey.isEmpty ? nil : apiKey
        )
        do {
            try await server.start(config)
            serverRunning = true
        } catch {
            logStore.append("Server failed to start: \(error)")
        }
    }

    public func stopServer() async {
        await server.stop()
        serverRunning = false
    }

    /// Start the server on launch if the user enabled auto-start. A model need not be loaded — the
    /// server runs and returns a `no_model_loaded` error per request until one is. Call once from the
    /// app's root `.task`. No-op if auto-start is off or the server is already running.
    public func autoStartServerIfEnabled() async {
        guard autoStartServer, !serverRunning else { return }
        logStore.append("Auto-starting server…")
        await startServer()
    }

    // MARK: - Playground (dogfood the local server)

    /// Send a chat message to the *running local server* and stream the reply text.
    /// Returns the full assistant text. Throws if the server isn't running.
    public func playgroundSend(_ prompt: String, onDelta: @escaping @MainActor (String) -> Void) async throws {
        guard serverRunning else {
            throw AppError.serverNotRunning
        }
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // Don't reuse pooled connections: a prior streamed reply can leave a keep-alive
        // connection in a state URLSession marks unusable, which surfaced as
        // "could not connect" on the *second* playground message.
        request.setValue("close", forHTTPHeaderField: "Connection")
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        let payload: [String: Any] = [
            "model": modelState.loadedDescriptor?.repoID ?? "local",
            "stream": true,
            "messages": [["role": "user", "content": prompt]],
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        // A fresh ephemeral session per request avoids cross-request connection pooling entirely.
        let config = URLSessionConfiguration.ephemeral
        config.httpMaximumConnectionsPerHost = 1
        let session = URLSession(configuration: config)
        defer { session.finishTasksAndInvalidate() }

        var sawDone = false
        let (bytes, _) = try await session.bytes(for: request)
        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6))
            if payload == "[DONE]" { sawDone = true; continue }   // drain to EOF, don't abandon
            if sawDone { continue }
            guard let data = payload.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = obj["choices"] as? [[String: Any]],
                  let delta = choices.first?["delta"] as? [String: Any],
                  let content = delta["content"] as? String
            else { continue }
            onDelta(content)
        }
    }

    public enum AppError: Error, LocalizedError {
        case serverNotRunning
        public var errorDescription: String? {
            switch self {
            case .serverNotRunning: "Start the server before using the playground."
            }
        }
    }

    // MARK: - Copilot config

    /// The VS Code chatLanguageModels.json entry for the currently-loaded model, if any.
    public func copilotConfigSnippet() -> String? {
        guard let descriptor = modelState.loadedDescriptor else { return nil }
        let caps = ModelCapabilityDetector.detect(repoID: descriptor.repoID)
        let effectiveHost = bindLAN ? host : "127.0.0.1"  // advertise loopback for the local case
        return CopilotConfig.modelEntry(
            repoID: descriptor.repoID,
            host: effectiveHost,
            port: port,
            capabilities: caps
        )
    }
}
