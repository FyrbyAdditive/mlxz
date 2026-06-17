import Foundation
import MLXZCore
import MLXLMCommon
import MLXLLM
import MLXVLM
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// Loads models from HuggingFace into `MLXInferenceEngine`s using mlx-swift-lm.
///
/// Importing both `MLXLLM` and `MLXVLM` registers their factories, so the free
/// `loadModelContainer` auto-dispatches between text and vision models — Qwen dense, Qwen
/// MoE (Qwen3.6-35B-A3B), and VLMs all load through this one path.
public struct MLXModelLoader: ModelLoading {
    private let perf: EnginePerfOptions

    public init(perf: EnginePerfOptions = .default) {
        self.perf = perf
    }

    public func load(
        _ descriptor: ModelDescriptor,
        progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws -> any InferenceEngine {
        let configuration = ModelConfiguration(
            id: descriptor.repoID,
            revision: descriptor.revision ?? "main"
        )

        let container = try await loadModelContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration
        ) { p in
            progress(LoadProgress(fraction: p.fractionCompleted, detail: p.localizedDescription))
        }

        return MLXInferenceEngine(
            descriptor: descriptor,
            capabilities: ModelCapabilityDetector.detect(repoID: descriptor.repoID),
            container: container,
            perf: perf
        )
    }
}
