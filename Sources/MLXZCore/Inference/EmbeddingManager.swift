import Foundation

/// Caches loaded embedding engines keyed by repo id, loading on first use. Embedding models are
/// small, so we keep the most-recently-used one resident (one slot, swap on a different model).
public actor EmbeddingManager {
    private let loader: any EmbeddingLoading
    private var current: (any EmbeddingEngine)?

    public init(loader: any EmbeddingLoading) {
        self.loader = loader
    }

    /// Embed `request.inputs` with `request.model`, loading the model if it isn't resident.
    public func embed(_ request: EmbeddingRequest) async throws -> EmbeddingResult {
        let engine = try await engine(for: request.model)
        return try await engine.embed(request)
    }

    private func engine(for repoID: String) async throws -> any EmbeddingEngine {
        if let current, current.descriptor.repoID == repoID {
            return current
        }
        let engine = try await loader.loadEmbedding(ModelDescriptor(repoID: repoID))
        current = engine
        return engine
    }
}
