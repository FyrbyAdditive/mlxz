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
    /// Perf options are read PER LOAD via this provider, so the GUI can change them (KV bits,
    /// prefix-cache slots, snapshot block) and have the next model load pick them up without
    /// rebuilding the loader/manager. The CLI passes a constant.
    private let perfProvider: @Sendable () -> EnginePerfOptions
    /// Default drafter (e.g. from `mlxz-serve --mtp-draft`). A per-load `draftModelID` overrides it.
    private let defaultDraftModelID: String?
    /// DSpark drafter policy: "auto" (default — attach the official drafter when the target
    /// is supported), "off" (never), or an explicit drafter repo id.
    private let dsparkDraft: String

    public init(
        perf: EnginePerfOptions = .default, draftModelID: String? = nil,
        dsparkDraft: String? = nil
    ) {
        self.init(perfProvider: { perf }, draftModelID: draftModelID, dsparkDraft: dsparkDraft)
    }

    /// Provider-based init: `perfProvider` is invoked on each `load` so settings changes apply to
    /// the next load. (The GUI uses this; the convenience `init(perf:)` wraps a constant.)
    public init(
        perfProvider: @escaping @Sendable () -> EnginePerfOptions, draftModelID: String? = nil,
        dsparkDraft: String? = nil
    ) {
        self.perfProvider = perfProvider
        self.defaultDraftModelID = draftModelID
        self.dsparkDraft = dsparkDraft ?? "auto"
        // Tune the MLX runtime once, before any model is loaded (GPU cache limit is a process-wide
        // one-time setting). `configure` is idempotent; uses the initial perf snapshot.
        MLXRuntime.configure(perf: perfProvider())
    }

    public func load(
        _ descriptor: ModelDescriptor,
        draftModelID: String?,
        progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws -> any InferenceEngine {
        let draftModelID = draftModelID ?? defaultDraftModelID
        let perf = perfProvider()  // read current settings (GUI may have changed them)
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
        // `trimUnsound`: whether the model's caches can NOT be soundly trimmed for prefix reuse
        // (any rotating/hybrid layer — Gemma-3/4, gpt-oss, or anything under --max-kv-size). Those
        // models use per-request snapshot copies, which makes their plain requests safe to
        // interleave via the fair PlainScheduler. Probed on a throwaway cache (no GPU work).
        let probeParameters: GenerateParameters = {
            var p = GenerateParameters()
            p.maxKVSize = perf.maxKVSize
            return p
        }()
        let (isVisionModel, isBatchable, trimUnsound) = await container.perform { context in
            let probe = context.model.newCache(parameters: probeParameters)
            return (
                context.model is any VLMModel,
                context.model is BatchableModel,
                !(canTrimPromptCache(probe) && MLXInferenceEngine.isSoundlyTrimmable(probe))
            )
        }
        var capabilities = ModelCapabilityDetector.detect(
            repoID: descriptor.repoID, hasVisionConfig: isVisionModel)

        // Read config.json `model_type` (e.g. "gpt_oss") to refine output-format detection. The
        // snapshot is already cached (the container just loaded it), so this is a local read.
        let modelType = await Self.readModelType(
            repoID: descriptor.repoID, revision: descriptor.revision ?? "main")

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

        // DSpark draft-model speculation: attach the official drafter (auto-resolved for
        // supported targets, or an explicit repo). Auto mode is best-effort — any failure
        // (offline, incompatible) logs and serves plain; an explicit repo failure throws.
        var dsparkRuntime: DSparkRuntimeBox? = nil
        if draftModelID == nil, dsparkDraft.lowercased() != "off" {
            let auto = dsparkDraft.lowercased() == "auto"
            let resolved = auto
                ? DrafterPairing.dsparkDrafterRepoID(forTarget: descriptor.repoID)
                : dsparkDraft
            if let drafterRepo = resolved {
                do {
                    dsparkRuntime = try await Self.attachDSpark(
                        container: container, drafterRepo: drafterRepo,
                        targetRepoID: descriptor.repoID, revision: descriptor.revision ?? "main",
                        perf: perf, progress: progress)
                    capabilities.insert(.speculative)
                } catch where auto {
                    progress(LoadProgress(
                        fraction: nil,
                        detail: "DSpark drafter unavailable (\(error)) — serving without speculation"))
                }
            }
        }

        return MLXInferenceEngine(
            descriptor: descriptor,
            capabilities: capabilities,
            container: container,
            perf: perf,
            isBatchable: isBatchable,
            modelType: modelType,
            trimUnsound: trimUnsound,
            dspark: dsparkRuntime
        )
    }

    /// Download + load a DSpark drafter and validate it against the loaded target model.
    /// Runs the weight load inside `container.perform` (the drafter lives with the model).
    private static func attachDSpark(
        container: ModelContainer,
        drafterRepo: String,
        targetRepoID: String,
        revision: String,
        perf: EnginePerfOptions,
        progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws -> DSparkRuntimeBox {
        progress(LoadProgress(fraction: nil, detail: "downloading DSpark drafter \(drafterRepo)…"))
        guard let repo = Repo.ID(rawValue: drafterRepo) else {
            throw DSparkAttachError.badRepoID(drafterRepo)
        }
        let dir = try await HubClient().downloadSnapshot(of: repo, revision: "main")

        // The drafter was trained against a specific target architecture: same hidden width,
        // same layer count (its target_layer_ids index into them), same vocab. Fail loud on
        // mismatch instead of decoding garbage context.
        let targetConfig = try await Self.readTargetDims(repoID: targetRepoID, revision: revision)

        progress(LoadProgress(fraction: nil, detail: "loading DSpark drafter…"))
        let box = try await container.perform { context -> DSparkRuntimeBox in
            guard context.model is any DSparkTargetModel else {
                throw DSparkAttachError.targetNotSupported(String(describing: type(of: context.model)))
            }
            let drafter = try DSparkDraftLoader.load(directory: dir, quantBits: 4)
            let c = drafter.config
            if let t = targetConfig {
                guard c.hiddenSize == t.hiddenSize, c.vocabularySize == t.vocabSize,
                    c.numTargetLayers == t.hiddenLayers,
                    c.targetLayerIds.allSatisfy({ $0 < t.hiddenLayers })
                else {
                    throw DSparkAttachError.drafterMismatch(
                        drafter: drafterRepo, target: targetRepoID)
                }
            }
            return DSparkRuntimeBox(
                drafter: drafter,
                blockCap: perf.dsparkBlockCap,
                confidenceThreshold: perf.dsparkConfidenceThreshold)
        }
        progress(LoadProgress(fraction: nil, detail: "DSpark speculative decoding ready"))
        return box
    }

    /// Target dims for drafter validation, from the (cached) snapshot's config.json.
    private static func readTargetDims(
        repoID: String, revision: String
    ) async throws -> (hiddenSize: Int, hiddenLayers: Int, vocabSize: Int)? {
        guard let repo = Repo.ID(rawValue: repoID),
            let dir = try? await HubClient().downloadSnapshot(of: repo, revision: revision),
            let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let hidden = obj["hidden_size"] as? Int,
            let layers = obj["num_hidden_layers"] as? Int,
            let vocab = obj["vocab_size"] as? Int
        else { return nil }
        return (hidden, layers, vocab)
    }

    /// Read `model_type` from a model's `config.json` (the snapshot is expected to be cached). Used to
    /// refine output-format detection. nil on any miss — the format selector falls back to the repo id.
    static func readModelType(repoID: String, revision: String) async -> String? {
        guard let repo = Repo.ID(rawValue: repoID),
              let dir = try? await HubClient().downloadSnapshot(of: repo, revision: revision)
        else { return nil }
        let configURL = dir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return obj["model_type"] as? String
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

    /// Heuristic: standalone *drafter* checkpoints cannot be loaded as a primary model —
    /// mlx-community's `…-MTP-<quant>` (MTP head only) and deepseek-ai's `dspark_*` (drafter
    /// network whose config declares a full model_type, so the factory would try — and fail —
    /// to load it as a backbone).
    static func looksLikeStandaloneDrafter(_ repoID: String) -> Bool {
        let name = (repoID.split(separator: "/").last.map(String.init) ?? repoID).lowercased()
        return name.contains("-mtp-") || name.hasSuffix("-mtp") || name.contains("-mtp")
            || DrafterPairing.isDSparkDrafter(repoID)
    }
}

enum DSparkAttachError: Error, CustomStringConvertible {
    case badRepoID(String)
    case targetNotSupported(String)
    case drafterMismatch(drafter: String, target: String)

    var description: String {
        switch self {
        case .badRepoID(let id):
            return "invalid DSpark drafter repo id: \(id)"
        case .targetNotSupported(let type):
            return "loaded model (\(type)) does not expose DSpark hidden-state taps"
        case .drafterMismatch(let drafter, let target):
            return "DSpark drafter \(drafter) was not trained for target \(target) "
                + "(hidden size / layer count / vocab mismatch)"
        }
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
