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
    /// Default drafter (e.g. from `mlxz-serve --mtp-draft`). A per-load `draftModelID` overrides it.
    private let defaultDraftModelID: String?

    public init(perf: EnginePerfOptions = .default, draftModelID: String? = nil) {
        self.perf = perf
        self.defaultDraftModelID = draftModelID
        // Tune the MLX runtime once, before any model is loaded. Both composition roots (the
        // headless server and the GUI app) build the loader at startup, so this is the natural
        // single point; `configure` is idempotent.
        MLXRuntime.configure(perf: perf)
    }

    public func load(
        _ descriptor: ModelDescriptor,
        draftModelID: String?,
        progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws -> any InferenceEngine {
        let draftModelID = draftModelID ?? defaultDraftModelID
        // A standalone MTP drafter checkpoint (e.g. "…-MTP-4bit") contains only the MTP head, not a
        // full backbone — it can't be loaded as a primary model, only attached to one via
        // `draftModelID`. Reject it early with a clear message instead of failing deep in MLX.
        if Self.looksLikeStandaloneDrafter(descriptor.repoID) {
            throw MTPLoadError.drafterNotStandalone(descriptor.repoID)
        }

        let configuration = ModelConfiguration(
            id: descriptor.repoID,
            revision: descriptor.revision ?? "main"
        )

        // Some models (e.g. Qwen3.6) ship their chat template as a standalone `chat_template.jinja`
        // and leave `chat_template` absent from tokenizer_config.json. swift-transformers only reads
        // the template from tokenizer_config.json, so without this fixup the tokenizer has NO
        // template: tools and the `<tool_call>` protocol never render into the prompt, the model can
        // only narrate ("I should call list_dir") instead of emitting a tool call, and the VS Code
        // agent stalls. Fold the .jinja into tokenizer_config.json BEFORE the container builds the
        // tokenizer so it picks the template up.
        await Self.ensureChatTemplateInConfig(
            repoID: descriptor.repoID, revision: descriptor.revision ?? "main", progress: progress)

        let container = try await loadModelContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration
        ) { p in
            progress(LoadProgress(fraction: p.fractionCompleted, detail: p.localizedDescription))
        }

        // Authoritative vision detection: VLM checkpoints load as `VLMModel` (regardless of repo
        // name), so query the loaded model rather than guessing from the id. Also probe whether the
        // model supports continuous batching (BatchableModel) — captured once here so the engine
        // doesn't re-probe per request.
        let (isVisionModel, isBatchable) = await container.perform { context in
            (context.model is any VLMModel, context.model is BatchableModel)
        }
        var capabilities = ModelCapabilityDetector.detect(
            repoID: descriptor.repoID, hasVisionConfig: isVisionModel)

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
            perf: perf,
            isBatchable: isBatchable
        )
    }

    /// If the model's `tokenizer_config.json` has no `chat_template` but the snapshot ships a
    /// standalone `chat_template.jinja`, fold the .jinja into the config so swift-transformers (which
    /// only reads the config's `chat_template`) applies it — enabling tool rendering. Idempotent and
    /// best-effort: any failure leaves the files untouched and load proceeds.
    static func ensureChatTemplateInConfig(
        repoID: String, revision: String,
        progress: @escaping @Sendable (LoadProgress) -> Void
    ) async {
        guard let repo = Repo.ID(rawValue: repoID) else { return }
        let dir: URL
        do {
            dir = try await HubClient().downloadSnapshot(of: repo, revision: revision)
        } catch {
            return  // not yet cached / offline — skip; the normal load path still runs.
        }
        let configURL = dir.appendingPathComponent("tokenizer_config.json")
        let jinjaURL = dir.appendingPathComponent("chat_template.jinja")
        let fm = FileManager.default
        guard fm.fileExists(atPath: configURL.path), fm.fileExists(atPath: jinjaURL.path),
            let configData = try? Data(contentsOf: configURL),
            var config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any]
        else { return }
        // Already has a usable template → nothing to do.
        if let existing = config["chat_template"] as? String, !existing.isEmpty { return }
        guard let template = try? String(contentsOf: jinjaURL, encoding: .utf8),
            !template.isEmpty
        else { return }
        config["chat_template"] = template
        guard
            let merged = try? JSONSerialization.data(
                withJSONObject: config, options: [.sortedKeys])
        else { return }
        try? merged.write(to: configURL, options: .atomic)
        progress(LoadProgress(fraction: nil, detail: "applied chat template for tool calling"))
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

    /// Heuristic: mlx-community publishes MTP *drafters* as `…-MTP-<quant>` checkpoints that hold
    /// only the MTP head and cannot be loaded as a standalone model.
    static func looksLikeStandaloneDrafter(_ repoID: String) -> Bool {
        let name = (repoID.split(separator: "/").last.map(String.init) ?? repoID).lowercased()
        return name.contains("-mtp-") || name.hasSuffix("-mtp") || name.contains("-mtp")
    }
}

enum MTPLoadError: Error, CustomStringConvertible {
    case wrongModelType(String)
    case drafterNotStandalone(String)
    var description: String {
        switch self {
        case .wrongModelType(let t):
            "MTP drafter attach: loaded model is \(t), expected Qwen35Model"
        case .drafterNotStandalone(let id):
            "\(id) is an MTP drafter (it contains only the speculative head, not a full model). "
                + "Load the full base model and attach this drafter with --mtp-draft \(id)."
        }
    }
}
