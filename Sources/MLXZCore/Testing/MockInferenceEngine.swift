import Foundation

/// A scriptable `InferenceEngine` for tests and headless development.
/// Yields a fixed sequence of events, optionally with a delay between them.
public struct MockInferenceEngine: InferenceEngine {
    public let descriptor: ModelDescriptor
    public let capabilities: ModelCapabilities
    private let events: [GenerationEvent]
    private let perEventDelay: Duration?

    public init(
        descriptor: ModelDescriptor = .init(repoID: "mock/model"),
        capabilities: ModelCapabilities = [.chat, .tools],
        events: [GenerationEvent],
        perEventDelay: Duration? = nil
    ) {
        self.descriptor = descriptor
        self.capabilities = capabilities
        self.events = events
        self.perEventDelay = perEventDelay
    }

    /// Builds an engine that streams the given text as word deltas, then completes.
    public init(
        descriptor: ModelDescriptor = .init(repoID: "mock/model"),
        capabilities: ModelCapabilities = [.chat, .tools],
        streamingText text: String
    ) {
        var evts: [GenerationEvent] = [.started(.init(modelID: descriptor.repoID))]
        let words = text.split(separator: " ", omittingEmptySubsequences: false)
        for (i, w) in words.enumerated() {
            evts.append(.textDelta(i == 0 ? String(w) : " " + w))
        }
        let completion = TokenUsage(promptTokens: 1, completionTokens: words.count, tokensPerSecond: nil)
        evts.append(.completed(.init(finishReason: .stop, usage: completion)))
        self.init(descriptor: descriptor, capabilities: capabilities, events: evts)
    }

    public func generate(_ request: GenerationRequest) async throws -> AsyncThrowingStream<GenerationEvent, Error> {
        let events = self.events
        let delay = self.perEventDelay
        return AsyncThrowingStream { continuation in
            let task = Task {
                for event in events {
                    if Task.isCancelled { break }
                    if let delay { try? await Task.sleep(for: delay) }
                    continuation.yield(event)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// A `ModelLoading` that returns a preconfigured engine, recording the descriptor it was asked for.
public actor MockModelLoading: ModelLoading {
    public private(set) var requestedDescriptors: [ModelDescriptor] = []
    public private(set) var requestedDrafters: [String?] = []
    private let makeEngine: @Sendable (ModelDescriptor) -> any InferenceEngine
    private let emitProgress: Bool

    public init(
        emitProgress: Bool = true,
        makeEngine: @escaping @Sendable (ModelDescriptor) -> any InferenceEngine = { d in
            MockInferenceEngine(descriptor: d, streamingText: "hello world")
        }
    ) {
        self.emitProgress = emitProgress
        self.makeEngine = makeEngine
    }

    public func load(
        _ descriptor: ModelDescriptor,
        draftModelID: String?,
        progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws -> any InferenceEngine {
        requestedDescriptors.append(descriptor)
        requestedDrafters.append(draftModelID)
        if emitProgress {
            progress(LoadProgress(fraction: 0.5, detail: "downloading"))
            progress(LoadProgress(fraction: 1.0, detail: "ready"))
        }
        return makeEngine(descriptor)
    }
}
