import Foundation
import Logging

/// Owns the set of currently-loaded models (keyed by repo id). Shared by the HTTP server
/// and the UI; both observe `states()`. Several models can be resident at once (RAM
/// permitting); requests are routed to a specific one by id via `engine(for:)`.
public actor ModelManager {
    /// An observable snapshot of every model's lifecycle state. A struct (not an enum)
    /// because multiple models can be loaded/loading simultaneously.
    public struct Snapshot: Sendable, Equatable {
        public struct Loaded: Sendable, Equatable {
            public var descriptor: ModelDescriptor
            /// The attached MTP drafter repo id, if one was attached at load.
            public var drafterID: String?
        }
        public struct Loading: Sendable, Equatable {
            public var descriptor: ModelDescriptor
            public var fraction: Double?
        }
        public struct Failed: Sendable, Equatable {
            public var descriptor: ModelDescriptor
            public var message: String
        }

        /// Loaded models, most-recently-loaded first.
        public var loaded: [Loaded] = []
        public var loading: [Loading] = []
        public var failed: [Failed] = []

        public init(loaded: [Loaded] = [], loading: [Loading] = [], failed: [Failed] = []) {
            self.loaded = loaded
            self.loading = loading
            self.failed = failed
        }

        // MARK: Convenience accessors

        public func isLoaded(_ repoID: String) -> Bool {
            loaded.contains { $0.descriptor.repoID == repoID }
        }
        /// The loading fraction for a repo id (`some` while loading, `nil` otherwise).
        public func loadingFraction(_ repoID: String) -> Double? {
            loading.first { $0.descriptor.repoID == repoID }?.fraction
        }
        public func isLoading(_ repoID: String) -> Bool {
            loading.contains { $0.descriptor.repoID == repoID }
        }
        public func failureMessage(_ repoID: String) -> String? {
            failed.first { $0.descriptor.repoID == repoID }?.message
        }
        /// The attached drafter id for a loaded base model, if any.
        public func drafterID(for repoID: String) -> String? {
            loaded.first { $0.descriptor.repoID == repoID }?.drafterID
        }

        // MARK: Back-compat single-value accessors (the most-recently-loaded model)

        public var loadedDescriptor: ModelDescriptor? { loaded.first?.descriptor }
        public var attachedDrafterID: String? { loaded.first?.drafterID }
    }

    private let loader: any ModelLoading
    private let logger: Logger

    private(set) public var snapshot = Snapshot()
    /// Loaded engines, keyed by `descriptor.repoID`.
    private var engines: [String: any InferenceEngine] = [:]
    /// Monotonic "last used/loaded" tick per repo id, for LRU eviction under memory pressure.
    private var lastUsed: [String: Int] = [:]
    private var tick = 0

    /// Continuations for observers subscribed via `states()`.
    private var observers: [UUID: AsyncStream<Snapshot>.Continuation] = [:]

    public init(loader: any ModelLoading, logger: Logger = Logger(label: "mlxz.model-manager")) {
        self.loader = loader
        self.logger = logger
    }

    /// The engine for a specific model id (exact match), bumping its recency. Returns nil if
    /// that id isn't loaded — the server routes strictly and 404s on a miss.
    public func engine(for modelID: String) -> (any InferenceEngine)? {
        guard let engine = engines[modelID] else { return nil }
        touch(modelID)
        return engine
    }

    /// The sole loaded engine when exactly one is loaded, else nil. Used by the single-model
    /// CLI / benchmark paths that don't route by id.
    public func currentEngine() -> (any InferenceEngine)? {
        engines.count == 1 ? engines.values.first : nil
    }

    /// Repo ids of all currently-loaded models (most-recently-loaded first).
    public func loadedModelIDs() -> [String] { snapshot.loaded.map(\.descriptor.repoID) }

    /// Load a model, ADDING it to the resident set (does not unload others). Idempotent when
    /// the same descriptor+drafter is already loaded. `draftModelID` attaches a matching MTP
    /// drafter.
    public func load(_ descriptor: ModelDescriptor, draftModelID: String? = nil) async throws {
        let repoID = descriptor.repoID
        if snapshot.isLoaded(repoID), snapshot.drafterID(for: repoID) == draftModelID {
            touch(repoID)
            return
        }

        setLoading(descriptor, fraction: nil)
        logger.info("loading model", metadata: ["model": .string(descriptor.id)])

        do {
            let loaded = try await loader.load(descriptor, draftModelID: draftModelID) {
                [weak self] progress in
                guard let self else { return }
                Task { await self.setLoading(descriptor, fraction: progress.fraction) }
            }
            engines[repoID] = loaded
            touch(repoID)
            finishLoading(repoID)
            setLoaded(descriptor, drafterID: draftModelID)
            logger.info("model loaded", metadata: ["model": .string(descriptor.id)])
        } catch {
            finishLoading(repoID)
            setFailed(descriptor, message: String(describing: error))
            logger.error("model load failed", metadata: [
                "model": .string(descriptor.id),
                "error": .string(String(describing: error)),
            ])
            throw error
        }
    }

    /// Unload one model by repo id, freeing its memory. No-op if it isn't loaded.
    public func unload(_ repoID: String) async {
        engines[repoID] = nil
        lastUsed[repoID] = nil
        snapshot.loaded.removeAll { $0.descriptor.repoID == repoID }
        emit()
    }

    /// Unload every loaded model.
    public func unloadAll() async {
        engines.removeAll()
        lastUsed.removeAll()
        snapshot.loaded.removeAll()
        emit()
    }

    /// Evict the least-recently-used loaded model (called in a loop under critical memory
    /// pressure). Returns the evicted repo id, or nil if nothing is loaded.
    @discardableResult
    public func evictLeastRecentlyUsed() async -> String? {
        guard let oldest = engines.keys.min(by: { (lastUsed[$0] ?? 0) < (lastUsed[$1] ?? 0) })
        else { return nil }
        await unload(oldest)
        return oldest
    }

    // MARK: - Observation

    /// A stream of snapshot changes. The current snapshot is yielded immediately on subscription.
    public func states() -> AsyncStream<Snapshot> {
        AsyncStream { continuation in
            let id = UUID()
            observers[id] = continuation
            continuation.yield(snapshot)
            continuation.onTermination = { [weak self] _ in
                Task { await self?.removeObserver(id) }
            }
        }
    }

    private func removeObserver(_ id: UUID) { observers[id] = nil }

    // MARK: - Snapshot mutation

    private func touch(_ repoID: String) { tick += 1; lastUsed[repoID] = tick }

    private func setLoading(_ descriptor: ModelDescriptor, fraction: Double?) {
        snapshot.failed.removeAll { $0.descriptor.repoID == descriptor.repoID }
        if let i = snapshot.loading.firstIndex(where: { $0.descriptor.repoID == descriptor.repoID }) {
            snapshot.loading[i].fraction = fraction
        } else {
            snapshot.loading.append(.init(descriptor: descriptor, fraction: fraction))
        }
        emit()
    }

    private func finishLoading(_ repoID: String) {
        snapshot.loading.removeAll { $0.descriptor.repoID == repoID }
    }

    private func setLoaded(_ descriptor: ModelDescriptor, drafterID: String?) {
        snapshot.loaded.removeAll { $0.descriptor.repoID == descriptor.repoID }
        snapshot.loaded.insert(.init(descriptor: descriptor, drafterID: drafterID), at: 0)
        emit()
    }

    private func setFailed(_ descriptor: ModelDescriptor, message: String) {
        snapshot.failed.removeAll { $0.descriptor.repoID == descriptor.repoID }
        snapshot.failed.append(.init(descriptor: descriptor, message: message))
        emit()
    }

    private func emit() {
        for continuation in observers.values { continuation.yield(snapshot) }
    }
}
