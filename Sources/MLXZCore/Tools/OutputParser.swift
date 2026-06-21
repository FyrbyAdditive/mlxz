import Foundation

/// A streaming parser that splits a model's raw token-text stream into the three channels mlxz
/// surfaces to clients: **reasoning** (chain-of-thought), **visible** answer text, and **tool calls**.
///
/// Different model families encode these differently — Qwen/Hermes uses `<think>` + `<tool_call>`
/// tags, gpt-oss uses the OpenAI "harmony" channel format (`<|channel|>analysis<|message|>…`), Mistral
/// uses `[TOOL_CALLS]…`, etc. A single `OutputParser` owns all three channels at once because some
/// formats (harmony) interleave them in one marker grammar — a two-stage "reasoning then tool calls"
/// pipe can't express that.
///
/// Contract (matching the existing `ThinkParser` / `ToolCallParser` structs it generalizes):
///   - feed deltas via `consume(_:)`; it returns whatever is now classifiable;
///   - it is **streaming-safe**: a marker split across chunk boundaries is buffered, never leaked as
///     visible/reasoning text;
///   - call `finish()` once at end-of-stream to flush any buffered remainder.
public protocol OutputParser: Sendable {
    mutating func consume(_ chunk: String) -> OutputParse
    mutating func finish() -> OutputParse
}

/// The classified output of one `OutputParser` step: any new reasoning, visible text, and completed
/// tool calls. Channels are independent — a single `consume` may yield any combination.
public struct OutputParse: Sendable, Equatable {
    public var visibleText: String
    public var reasoning: String
    public var toolCalls: [ToolCall]

    public init(visibleText: String = "", reasoning: String = "", toolCalls: [ToolCall] = []) {
        self.visibleText = visibleText
        self.reasoning = reasoning
        self.toolCalls = toolCalls
    }

    /// Whether nothing was produced this step (lets callers skip empty yields).
    public var isEmpty: Bool {
        visibleText.isEmpty && reasoning.isEmpty && toolCalls.isEmpty
    }
}

/// The output format a model uses for reasoning + tool calls. Selected per-model by
/// `OutputParserFactory.detectFormat`; drives which `OutputParser` is built and a couple of
/// format-specific prompt hooks.
public enum OutputFormat: Sendable, Equatable, CaseIterable {
    /// Qwen / Hermes: `<think>…</think>` reasoning + `<tool_call>{json}</tool_call>` or
    /// `<tool_call><function=…>` XML calls. The mlxz default — everything not otherwise recognized.
    case qwenHermes
    /// gpt-oss OpenAI "harmony" channel format: `<|channel|>analysis|final|commentary…<|message|>…`.
    case harmony
    /// Mistral V11+: `[TOOL_CALLS]name [ARGS]{json}` (no reasoning channel).
    case mistral
    /// Llama 3 inline: `<|python_tag|>{ "name": …, "parameters": … }` (no reasoning channel).
    case llama3
    /// GLM4: `name<arg_key>k</arg_key><arg_value>v</arg_value>…` (no reasoning channel).
    case glm4
    /// Gemma: `<|tool_call>call:name{key:<|"|>v<|"|>}<tool_call|>` (no reasoning channel).
    case gemma

    /// Whether the model's chat template PRE-OPENS a `<think>` block in the prompt (so the stream
    /// starts inside reasoning and only emits the closing tag). Only Qwen/Hermes does this; harmony
    /// emits an explicit `<|channel|>analysis` instead.
    public var prefersPreOpenedThink: Bool { self == .qwenHermes }

    /// Whether the model's chat template understands the `enable_thinking` kwarg. Qwen and Gemma both
    /// gate reasoning on it (Gemma's template injects a `<|think|>` token when it's true).
    public var supportsEnableThinkingKwarg: Bool { self == .qwenHermes || self == .gemma }

    /// Whether this format carries reasoning in a dedicated channel that should surface as
    /// `reasoning_content` WHEN thinking is enabled. Gemma's `thought` channel is reasoning only in
    /// that case (with thinking off it's the visible answer — see `GemmaOutputParser`). Qwen/harmony
    /// handle reasoning via their own parsers, so this flag is Gemma-specific.
    public var thoughtChannelIsReasoningWhenThinking: Bool { self == .gemma }
}

/// Selects and builds the right `OutputParser` for a model. Pure and dependency-free (lives in
/// MLXZCore next to the parsers), mirroring `ModelCapabilityDetector`'s lowercased-haystack idiom.
///
/// Adding a new format is: implement an `OutputParser`, add one `make` branch, and one `detectFormat`
/// marker check — no engine changes.
public enum OutputParserFactory {
    /// Pick the output format from a model's identity. Uses the same `(repoID + " " + modelType)`
    /// lowercased haystack as `ModelCapabilityDetector`. Unknown → `.qwenHermes` (the safe default, so
    /// existing models never regress). `capabilities` is accepted for future format signals.
    public static func detectFormat(
        repoID: String,
        modelType: String? = nil,
        capabilities: ModelCapabilities = []
    ) -> OutputFormat {
        let haystack = (repoID + " " + (modelType ?? "")).lowercased()

        // gpt-oss → harmony. Markers mirror `usesAttentionSinks` so any repo recognized as gpt-oss
        // also gets harmony parsing.
        if ["gpt-oss", "gpt_oss", "gptoss", "harmony"].contains(where: haystack.contains) {
            return .harmony
        }
        // Mistral V11+ XML/[TOOL_CALLS]. The fork maps `mistral3*` model types to this format.
        if haystack.contains("mistral3") || haystack.contains("ministral") {
            return .mistral
        }
        // GLM4 family (glm4, glm4_moe, …).
        if haystack.contains("glm4") || haystack.contains("glm-4") {
            return .glm4
        }
        // Gemma family (gemma, gemma-3, gemma-4, …).
        if haystack.contains("gemma") {
            return .gemma
        }
        // Llama 3+ inline `<|python_tag|>`. Match the family but avoid older Llama 1/2; the `3`/`4`
        // markers in the repo id or model type are the practical signal for current MLX conversions.
        if haystack.contains("llama-3") || haystack.contains("llama3")
            || haystack.contains("llama-4") || haystack.contains("llama4") {
            return .llama3
        }
        return .qwenHermes
    }

    /// Build a fresh parser for `format`.
    /// - Parameters:
    ///   - startInsideThink: honored only by formats that pre-open a think block (`qwenHermes`).
    ///   - thinkingEnabled: whether reasoning is on for this request. Gemma routes its `thought`
    ///     channel to reasoning only when thinking is enabled (otherwise it's the visible answer).
    public static func make(
        format: OutputFormat, startInsideThink: Bool, thinkingEnabled: Bool = false
    ) -> any OutputParser {
        switch format {
        case .qwenHermes: return QwenOutputParser(startInsideThink: startInsideThink)
        case .harmony:    return HarmonyOutputParser()
        case .mistral:    return MistralOutputParser()
        case .llama3:     return Llama3OutputParser()
        case .glm4:       return GLM4OutputParser()
        case .gemma:
            return GemmaOutputParser(
                thoughtIsReasoning: format.thoughtChannelIsReasoningWhenThinking && thinkingEnabled)
        }
    }
}
