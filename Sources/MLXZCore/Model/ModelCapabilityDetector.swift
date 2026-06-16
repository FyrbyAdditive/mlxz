import Foundation

/// Infers model capabilities from a HuggingFace repo id (and optionally a `model_type` from
/// config.json). Pure and dependency-free so it is unit-testable without loading a model.
///
/// Chat + tools are always advertised — we own tool-call parsing, so every chat model can be
/// driven in Copilot agent mode. Vision and speculative decoding are inferred from naming.
public enum ModelCapabilityDetector {
    public static func detect(repoID: String, modelType: String? = nil) -> ModelCapabilities {
        var caps: ModelCapabilities = [.chat, .tools]
        let haystack = (repoID + " " + (modelType ?? "")).lowercased()

        if isVision(haystack) {
            caps.insert(.vision)
        }
        if isSpeculative(haystack) {
            caps.insert(.speculative)
        }
        return caps
    }

    private static func isVision(_ s: String) -> Bool {
        // Vision-language naming conventions in the mlx-community catalog.
        let markers = ["-vl", "_vl", "vl-", "vision", "llava", "paligemma",
                       "idefics", "smolvlm", "pixtral", "internvl", "-vlm"]
        return markers.contains { s.contains($0) }
    }

    private static func isSpeculative(_ s: String) -> Bool {
        // MTP models carry built-in multi-token-prediction heads (config model_type qwen3_5_mtp).
        s.contains("mtp")
    }
}
