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
    public private(set) var modelState: ModelManager.Snapshot = .init()
    public private(set) var serverRunning: Bool = false
    /// Per-loaded-model active speculative-decoding mode ("DSpark drafter: …", "native
    /// MTP"), keyed by repo id; absent when plain. Drives the ⚡ badge per model.
    public private(set) var speculationStatus: [String: String] = [:]
    /// A pending "this may exceed your RAM" confirmation for a load the user must approve.
    public var pendingRAMConfirm: RAMConfirm? = nil

    public struct RAMConfirm: Identifiable, Sendable {
        public let id = UUID()
        public let descriptor: ModelDescriptor
        public let message: String
    }

    // Live server metrics (an observable holder so the server's sink can update it without
    // capturing `self` during init).
    public let metrics = ServerMetrics()
    public var requestsServed: Int { metrics.requestsServed }
    public var lastTokensPerSecond: Double? { metrics.lastTokensPerSecond }

    public let logStore = LogStore()
    public let catalog = HubCatalog()
    public let localStore = LocalModelStore()
    public let downloads: DownloadManager

    // MARK: - Model library / Discover state
    //
    // Hoisted OUT of the view so it survives tab switches (it used to be view-local `@State`
    // that reset every time the user left the Models tab). All @MainActor.

    /// Cached installed models, refreshed on appear and whenever a download completes.
    public private(set) var installed: [InstalledModel] = []
    /// Latest Discover search results (already drafter-filtered/sorted by the catalog).
    public private(set) var discoverResults: [CatalogEntry] = []
    /// The live search text. Setting it schedules a debounced search.
    public var searchQuery: String = ""
    /// Restrict Discover to the curated mlx-community org, or search all MLX authors.
    public var mlxCommunityOnly: Bool = true
    public var sortKey: CatalogSort = .popular
    public var facet: ModelFacet = .all
    public private(set) var isSearching: Bool = false
    /// A human-readable search error (offline / HTTP), surfaced instead of a blank list.
    public private(set) var searchError: String? = nil

    private var searchTask: Task<Void, Never>?

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
        self.server = InferenceServer(
            manager: manager,
            logSink: { line in
                Task { @MainActor in logStore.append(line) }
            },
            embeddingManager: EmbeddingManager(loader: embeddingLoader),
            metricsSink: { [metrics] usage in
                Task { @MainActor in metrics.record(usage) }
            }
        )

        // Evict the least-recently-used model on critical memory pressure to avoid an OS
        // kill, keeping the most-recently-used (likely in-use) one.
        memoryMonitor = MemoryPressureMonitor { [weak self] isCritical in
            guard isCritical else { return }
            Task { @MainActor in
                guard let self, self.autoUnloadOnMemoryPressure,
                      !self.modelState.loaded.isEmpty else { return }
                if let evicted = await self.manager.evictLeastRecentlyUsed() {
                    self.speculationStatus[evicted] = nil
                    self.logStore.append("⚠️ Critical memory pressure — evicted \(evicted).")
                }
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

    /// Refresh the cached `installed` list (call on appear + after a download completes).
    public func refreshInstalled() {
        installed = localStore.installedModels()
    }

    // MARK: - Discover (HuggingFace search)

    /// Schedule a debounced search for the current `searchQuery`/`sortKey`/author scope.
    /// Cancels any pending/in-flight query first. An empty query returns HF trending.
    public func scheduleSearch(debounce: Duration = .milliseconds(350)) {
        searchTask?.cancel()
        let query = searchQuery
        let sort = sortKey
        let author = mlxCommunityOnly ? "mlx-community" : nil
        searchTask = Task { [weak self] in
            try? await Task.sleep(for: debounce)
            guard !Task.isCancelled else { return }
            await self?.runSearch(query: query, author: author, sort: sort)
        }
    }

    /// Run a search immediately (used by the explicit submit + Retry). Surfaces errors into
    /// `searchError` instead of collapsing them to an empty list.
    public func runSearch(query: String, author: String?, sort: CatalogSort) async {
        isSearching = true
        searchError = nil
        defer { isSearching = false }
        do {
            let entries = try await catalog.search(query: query, author: author, sort: sort)
            guard !Task.isCancelled else { return }
            discoverResults = entries
        } catch is CancellationError {
            // superseded by a newer query — leave state alone
        } catch {
            discoverResults = []
            searchError = "Couldn’t reach HuggingFace. Check your connection and try again."
        }
    }

    /// Kick off the initial Discover load (trending) if nothing has been searched yet.
    public func loadInitialDiscoverIfNeeded() async {
        guard discoverResults.isEmpty, searchError == nil, !isSearching else { return }
        await runSearch(query: "", author: mlxCommunityOnly ? "mlx-community" : nil, sort: sortKey)
    }

    // MARK: - Unified per-model status

    /// The single reconciled status a card renders from (see `ModelStatus`).
    public func status(of repoID: String) -> ModelStatus {
        let loaded = modelState.isLoaded(repoID)
        let loading = modelState.isLoading(repoID)
        let isDrafter = DrafterPairing.isDrafter(repoID)
        // A drafter is "attached" when it's the drafter of any loaded base model.
        let attached = modelState.loaded.contains { $0.drafterID == repoID }
        var fraction: Double? = nil
        var failed = false
        if let dl = downloads.downloads.first(where: { $0.id == repoID }) {
            switch dl.state {
            case .downloading(let f, _, _): fraction = f
            case .failed: failed = true
            case .done, .cancelled: break
            }
        }
        let onDisk = installed.contains { $0.descriptor.repoID == repoID }
        return ModelStatus.resolve(
            loaded: loaded, loading: loading, isDrafter: isDrafter, attachedDrafter: attached,
            downloadFraction: fraction, downloadFailed: failed, installed: onDisk)
    }

    /// Delete an installed model from the local cache. If it's the currently-loaded model, unload it
    /// first so we don't yank files out from under the running engine.
    public func deleteModel(_ model: InstalledModel) async {
        if modelState.isLoaded(model.descriptor.repoID) {
            await unload(model.descriptor.repoID)
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

    /// Load a model, ADDING it to the resident set (other loaded models stay). If it would
    /// push total weights past a comfortable share of RAM, stash a confirmation instead of
    /// loading; the UI presents it and calls `confirmLoad` to proceed.
    public func load(_ descriptor: ModelDescriptor) async {
        if modelState.isLoaded(descriptor.repoID) { return }  // already resident
        if let message = ramWarning(for: descriptor.repoID) {
            pendingRAMConfirm = RAMConfirm(descriptor: descriptor, message: message)
            return
        }
        await performLoad(descriptor)
    }

    /// Proceed with a load the user approved despite the RAM warning.
    public func confirmLoad(_ descriptor: ModelDescriptor) async {
        pendingRAMConfirm = nil
        await performLoad(descriptor)
    }

    private func performLoad(_ descriptor: ModelDescriptor) async {
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
            let speculation = await manager.engine(for: descriptor.repoID)?.speculationStatus
            speculationStatus[descriptor.repoID] = speculation
            if let speculation {
                logStore.append("Loaded \(descriptor.repoID) — speculative decoding ON (\(speculation))")
            } else {
                logStore.append("Loaded \(descriptor.repoID)")
            }
        } catch {
            speculationStatus[descriptor.repoID] = nil
            logStore.append("Load failed: \(error)")
        }
    }

    /// A warning string if loading `repoID` alongside the already-loaded models would strain
    /// RAM (`.tight`/`.exceeds`), else nil. Sizes come from the local cache; unknown sizes
    /// don't warn.
    private func ramWarning(for repoID: String) -> String? {
        let sizeOf: (String) -> Int64? = { id in
            self.installed.first { $0.descriptor.repoID == id }?.sizeBytes
        }
        guard let newSize = sizeOf(repoID) else { return nil }
        let loadedSize = modelState.loaded.reduce(Int64(0)) { $0 + (sizeOf($1.descriptor.repoID) ?? 0) }
        let total = loadedSize + newSize
        switch RAMFit.of(sizeBytes: total) {
        case .fits, .unknown: return nil
        case .tight, .exceeds:
            let ram = Int64(ProcessInfo.processInfo.physicalMemory)
            return "\(ByteFormat.string(loadedSize)) already loaded + "
                + "\(ByteFormat.string(newSize)) new ≈ \(ByteFormat.string(total)) "
                + "of \(ByteFormat.string(ram)) RAM. This may be slow or get evicted. Load anyway?"
        }
    }

    /// The repo id of an installed MTP drafter that pairs with `baseRepoID`, if one is present.
    public func matchingInstalledDrafter(for baseRepoID: String) -> String? {
        guard !DrafterPairing.isDrafter(baseRepoID) else { return nil }
        let expected = DrafterPairing.drafterRepoID(forBase: baseRepoID)
        return installedModels().first { $0.descriptor.repoID == expected }?.descriptor.repoID
    }

    /// Unload one model by repo id (others stay resident).
    public func unload(_ repoID: String) async {
        await manager.unload(repoID)
        speculationStatus[repoID] = nil
        logStore.append("Unloaded \(repoID)")
    }

    /// Unload every loaded model.
    public func unloadAll() async {
        await manager.unloadAll()
        speculationStatus.removeAll()
        logStore.append("Unloaded all models")
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
    /// Send a chat message to the running server. `modelID` selects which loaded model
    /// answers (routing is strict now); defaults to the most-recently-loaded model.
    public func playgroundSend(_ prompt: String, model modelID: String? = nil, onDelta: @escaping @MainActor (String) -> Void) async throws {
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
            "model": modelID ?? modelState.loadedDescriptor?.repoID ?? "local",
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

    /// The VS Code chatLanguageModels.json entry for a loaded model (defaults to the
    /// most-recently-loaded), if any.
    public func copilotConfigSnippet(for repoID: String? = nil) -> String? {
        let descriptor = repoID.map { ModelDescriptor(repoID: $0) } ?? modelState.loadedDescriptor
        guard let descriptor else { return nil }
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
