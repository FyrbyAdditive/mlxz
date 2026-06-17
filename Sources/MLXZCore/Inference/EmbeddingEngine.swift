import Foundation

/// A request to embed one or more texts.
public struct EmbeddingRequest: Sendable {
    public var inputs: [String]
    public var model: String
    public init(inputs: [String], model: String) {
        self.inputs = inputs
        self.model = model
    }
}

/// The result of embedding: one vector per input, plus token accounting.
public struct EmbeddingResult: Sendable {
    public var vectors: [[Float]]
    public var promptTokens: Int
    public init(vectors: [[Float]], promptTokens: Int = 0) {
        self.vectors = vectors
        self.promptTokens = promptTokens
    }
}

/// Produces embedding vectors for text. Independent of the chat `InferenceEngine` — embedding
/// models are a different model type loaded into a separate slot.
public protocol EmbeddingEngine: Sendable {
    var descriptor: ModelDescriptor { get }
    func embed(_ request: EmbeddingRequest) async throws -> EmbeddingResult
}

/// Loads an embedding model by descriptor (downloads on demand). Injected at the composition root.
public protocol EmbeddingLoading: Sendable {
    func loadEmbedding(_ descriptor: ModelDescriptor) async throws -> any EmbeddingEngine
}
