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
    // Server config (bound to the UI).
    public var host: String = "127.0.0.1"
    public var port: Int = 8080
    public var bindLAN: Bool = false
    public var apiKey: String = ""

    // Observable state.
    public private(set) var modelState: ModelManager.State = .empty
    public private(set) var serverRunning: Bool = false

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

    public init(
        loader: any ModelLoading,
        downloader: any ModelDownloading,
        embeddingLoader: any EmbeddingLoading
    ) {
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

    // MARK: - Downloads

    public func startDownload(_ repoID: String) {
        logStore.append("Downloading \(repoID)…")
        downloads.start(repoID)
    }

    public func cancelDownload(_ repoID: String) {
        downloads.cancel(repoID)
    }

    // MARK: - Model lifecycle

    public func load(_ descriptor: ModelDescriptor) async {
        logStore.append("Loading \(descriptor.repoID)…")
        do {
            try await manager.load(descriptor)
            logStore.append("Loaded \(descriptor.repoID)")
        } catch {
            logStore.append("Load failed: \(error)")
        }
    }

    public func unload() async {
        await manager.unload()
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
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        let payload: [String: Any] = [
            "model": modelState.loadedDescriptor?.repoID ?? "local",
            "stream": true,
            "messages": [["role": "user", "content": prompt]],
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (bytes, _) = try await URLSession.shared.bytes(for: request)
        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6))
            if payload == "[DONE]" { break }
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
