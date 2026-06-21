import Foundation

/// Descriptive, human-readable labels for a model — its tuning (instruct/base/…), architecture
/// (MoE/vision/…), and quantization — inferred from the repo id and optional `model_type`. Pure and
/// dependency-free so it's unit-testable and shared by the search + installed lists.
///
/// These are presentation hints, not capabilities (`ModelCapabilities` drives request validation).
public enum ModelTypeInfo {
    /// How the model was fine-tuned, inferred from naming. `nil` when there's no clear signal.
    public enum Tuning: String, Sendable {
        case instruct = "Instruct"   // chat/instruction-tuned (…-it, -instruct, -chat)
        case reasoning = "Reasoning" // explicit reasoning/thinking variants
        case base = "Base"           // pretrained base (no instruct tuning) — explicitly marked
        case code = "Code"           // code-specialized
    }

    /// The tuning label for a model, or nil if it can't be inferred confidently.
    public static func tuning(repoID: String, modelType: String? = nil) -> Tuning? {
        let s = (repoID + " " + (modelType ?? "")).lowercased()
        // Reasoning takes precedence (a reasoning model is also instruct-y, but the reasoning label
        // is the more informative one).
        if s.contains("reason") || s.contains("thinking") || s.contains("-r1") || s.contains("deepseek-r")
            || s.contains("qwq") {
            return .reasoning
        }
        if s.contains("code") || s.contains("coder") || s.contains("starcoder") || s.contains("codestral") {
            return .code
        }
        // Instruct/chat markers (tokenized so we don't match "it" inside other words).
        let instructMarkers = ["-it", "_it", "-it-", "instruct", "-chat", "_chat", "-sft", "-dpo", "-rl"]
        if instructMarkers.contains(where: { s.contains($0) }) || s.hasSuffix("-it") {
            return .instruct
        }
        // Explicit base markers — only label "Base" when stated, to avoid mislabeling an instruct
        // model whose id simply omits "-it".
        if s.contains("-base") || s.contains("_base") || s.contains("-pt") || s.hasSuffix("base") {
            return .base
        }
        return nil
    }

    /// Whether the model is a mixture-of-experts model (MoE) — denser to label than infer from caps.
    public static func isMoE(repoID: String, modelType: String? = nil) -> Bool {
        let s = (repoID + " " + (modelType ?? "")).lowercased()
        if s.contains("moe") || s.contains("mixtral") { return true }
        // "a3b" / "a22b" active-expert naming, or "8x7b" / "8x22b" expert-count naming.
        if s.range(of: #"a\d+b"#, options: .regularExpression) != nil { return true }
        if s.range(of: #"\d+x\d+b"#, options: .regularExpression) != nil { return true }
        return false
    }

    /// Quantization label parsed from the id (e.g. "4-bit", "8-bit", "BF16"), or nil if not present.
    public static func quantization(repoID: String) -> String? {
        let lower = repoID.lowercased()
        for q in ["2bit", "3bit", "4bit", "5bit", "6bit", "8bit"] where lower.contains(q) {
            return q.replacingOccurrences(of: "bit", with: "-bit")
        }
        for q in ["bf16", "fp16", "f16", "mxfp4", "fp8", "f8"] where lower.contains(q) {
            return q.uppercased()
        }
        return nil
    }
}
