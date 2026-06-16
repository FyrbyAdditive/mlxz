import Foundation
import Logging

/// Owns the single loaded model. Shared by the HTTP server and the UI; both observe `states()`.
public actor ModelManager {
    public enum State: Sendable {
        case empty
        case loading(ModelDescriptor, fraction: Double?)
        case loaded(ModelDescriptor)
        case failed(ModelDescriptor, message: String)

        public var loadedDescriptor: ModelDescriptor? {
            if case let .loaded(d) = self { return d }
            return nil
        }
    }

    private let loader: any ModelLoading
    private let logger: Logger

    private(set) public var state: State = .empty
    private var engine: (any InferenceEngine)?

    /// Continuations for observers subscribed via `states()`.
    private var observers: [UUID: AsyncStream<State>.Continuation] = [:]

    public init(loader: any ModelLoading, logger: Logger = Logger(label: "mlxz.model-manager")) {
        self.loader = loader
        self.logger = logger
    }

    /// The currently loaded engine, if any. The server reads this per request.
    public func currentEngine() -> (any InferenceEngine)? { engine }

    /// Load a model, replacing any currently loaded one. Idempotent for an already-loaded descriptor.
    public func load(_ descriptor: ModelDescriptor) async throws {
        if case let .loaded(current) = state, current == descriptor {
            return
        }
        // Free the old model before loading a new one to avoid double memory pressure.
        await unload()

        update(.loading(descriptor, fraction: nil))
        logger.info("loading model", metadata: ["model": .string(descriptor.id)])

        do {
            let loaded = try await loader.load(descriptor) { [weak self] progress in
                guard let self else { return }
                Task { await self.update(.loading(descriptor, fraction: progress.fraction)) }
            }
            self.engine = loaded
            update(.loaded(descriptor))
            logger.info("model loaded", metadata: ["model": .string(descriptor.id)])
        } catch {
            update(.failed(descriptor, message: String(describing: error)))
            logger.error("model load failed", metadata: [
                "model": .string(descriptor.id),
                "error": .string(String(describing: error)),
            ])
            throw error
        }
    }

    /// Drop the loaded model and free its memory.
    public func unload() async {
        engine = nil
        update(.empty)
    }

    // MARK: - Observation

    /// A stream of state changes. The current state is yielded immediately on subscription.
    public func states() -> AsyncStream<State> {
        AsyncStream { continuation in
            let id = UUID()
            observers[id] = continuation
            continuation.yield(state)
            continuation.onTermination = { [weak self] _ in
                Task { await self?.removeObserver(id) }
            }
        }
    }

    private func removeObserver(_ id: UUID) {
        observers[id] = nil
    }

    private func update(_ newState: State) {
        state = newState
        for continuation in observers.values {
            continuation.yield(newState)
        }
    }
}
