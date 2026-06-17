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
    /// Optional standalone MTP drafter repo id to attach to the backbone (draft-model MTP).
    private let draftModelID: String?

    public init(perf: EnginePerfOptions = .default, draftModelID: String? = nil) {
        self.perf = perf
        self.draftModelID = draftModelID
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

        var capabilities = ModelCapabilityDetector.detect(repoID: descriptor.repoID)

        // Draft-model MTP: download the standalone drafter and attach it to the backbone's MTP head.
        if let draftModelID {
            progress(LoadProgress(fraction: nil, detail: "downloading MTP drafter \(draftModelID)…"))
            let hub = HubClient()
            guard let repo = Repo.ID(rawValue: draftModelID) else {
                throw MTPDraftError.noWeights
            }
            let drafterDir = try await hub.downloadSnapshot(of: repo, revision: "main")
            let quant = try Self.readQuantization(drafterDir)
            try await container.perform { context in
                guard let model = context.model as? any MTPSpeculativeModel else {
                    throw MTPLoadError.wrongModelType(String(describing: type(of: context.model)))
                }
                try MTPDraftLoader.attach(
                    to: model, drafterDirectory: drafterDir, quantization: quant)
            }
            capabilities.insert(.speculative)
        }

        return MLXInferenceEngine(
            descriptor: descriptor,
            capabilities: capabilities,
            container: container,
            perf: perf
        )
    }

    /// Read the drafter's quantization (bits/group size) from its config.json.
    private static func readQuantization(_ dir: URL) throws -> BaseConfiguration.Quantization? {
        let configURL = dir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let q = (obj["quantization"] ?? obj["quantization_config"]) as? [String: Any],
              let bits = q["bits"] as? Int,
              let groupSize = q["group_size"] as? Int
        else { return nil }
        return BaseConfiguration.Quantization(groupSize: groupSize, bits: bits)
    }
}

enum MTPLoadError: Error, CustomStringConvertible {
    case wrongModelType(String)
    var description: String {
        switch self {
        case .wrongModelType(let t): "MTP drafter attach: loaded model is \(t), expected Qwen35Model"
        }
    }
}
