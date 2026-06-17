import Foundation

/// A deterministic embedding engine for tests: returns a fixed-dim vector derived from input length.
public struct MockEmbeddingEngine: EmbeddingEngine {
    public let descriptor: ModelDescriptor
    private let dimension: Int

    public init(descriptor: ModelDescriptor = .init(repoID: "mock/embed"), dimension: Int = 4) {
        self.descriptor = descriptor
        self.dimension = dimension
    }

    public func embed(_ request: EmbeddingRequest) async throws -> EmbeddingResult {
        let vectors = request.inputs.map { text -> [Float] in
            (0..<dimension).map { i in Float((text.count + i) % 7) / 7.0 }
        }
        return EmbeddingResult(vectors: vectors, promptTokens: request.inputs.reduce(0) { $0 + $1.count })
    }
}

/// A `ModelLoading`-equivalent for embeddings, returning a mock engine.
public struct MockEmbeddingLoading: EmbeddingLoading {
    public init() {}
    public func loadEmbedding(_ descriptor: ModelDescriptor) async throws -> any EmbeddingEngine {
        MockEmbeddingEngine(descriptor: descriptor)
    }
}
