import Foundation
import MLXZCore
import MLX
import MLXEmbedders
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// Loads embedding models from HuggingFace via MLXEmbedders.
public struct MLXEmbeddingLoader: EmbeddingLoading {
    public init() {}

    public func loadEmbedding(_ descriptor: ModelDescriptor) async throws -> any EmbeddingEngine {
        let configuration = ModelConfiguration(id: descriptor.repoID, revision: descriptor.revision ?? "main")
        let container = try await EmbedderModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration
        )
        return MLXEmbeddingEngine(descriptor: descriptor, container: container)
    }
}

/// An `EmbeddingEngine` backed by an MLXEmbedders `EmbedderModelContainer`.
public struct MLXEmbeddingEngine: EmbeddingEngine {
    public let descriptor: ModelDescriptor
    private let container: EmbedderModelContainer

    init(descriptor: ModelDescriptor, container: EmbedderModelContainer) {
        self.descriptor = descriptor
        self.container = container
    }

    public func embed(_ request: EmbeddingRequest) async throws -> EmbeddingResult {
        let texts = request.inputs
        let result: (vectors: [[Float]], tokenCount: Int) = await container.perform { context in
            let model = context.model
            let tokenizer = context.tokenizer
            let pooling = context.pooling

            let encoded = texts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
            let tokenCount = encoded.reduce(0) { $0 + $1.count }
            let maxLength = encoded.reduce(into: 16) { $0 = max($0, $1.count) }
            let eos = tokenizer.eosTokenId ?? 0

            let padded = stacked(encoded.map { ids in
                MLXArray(ids + Array(repeating: eos, count: maxLength - ids.count))
            })
            let mask = padded .!= eos
            let tokenTypes = MLXArray.zeros(like: padded)

            let pooled = pooling(
                model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
                mask: mask,
                normalize: true,
                applyLayerNorm: true
            )
            pooled.eval()
            return (pooled.map { $0.asArray(Float.self) }, tokenCount)
        }

        return EmbeddingResult(vectors: result.vectors, promptTokens: result.tokenCount)
    }
}
