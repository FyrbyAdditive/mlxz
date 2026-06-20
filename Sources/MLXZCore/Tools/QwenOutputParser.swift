import Foundation

/// `OutputParser` for Qwen / Hermes models — the mlxz default.
///
/// This is a thin adapter that composes the two existing, battle-tested parsers in the exact order
/// the engine used before the `OutputParser` abstraction existed:
///   1. `ThinkParser` splits `<think>…</think>` reasoning from the visible stream;
///   2. `ToolCallParser` extracts `<tool_call>{json}</tool_call>` / `<function=…>` XML calls from the
///      visible portion.
///
/// Both inner parsers are kept UNCHANGED (their test suites encode hard-won streaming bugs — pre-opened
/// think blocks, partial-tag buffering, cross-request id uniqueness, Qwen XML calls). Reproducing the
/// engine's old `consume`/`finish` logic here verbatim makes Qwen output byte-identical, which is the
/// zero-regression guarantee.
public struct QwenOutputParser: OutputParser {
    private var think: ThinkParser
    private var tools: ToolCallParser

    /// - Parameters:
    ///   - startInsideThink: begin inside the reasoning block (the Qwen chat template pre-opens
    ///     `<think>` so the stream starts in reasoning and only emits the closing tag).
    ///   - idPrefix: tool-call id prefix (defaults to a process-unique token, as `ToolCallParser` does).
    public init(startInsideThink: Bool, idPrefix: String? = nil) {
        self.think = ThinkParser(startInsideThink: startInsideThink)
        self.tools = ToolCallParser(idPrefix: idPrefix)
    }

    public mutating func consume(_ chunk: String) -> OutputParse {
        let split = think.consume(chunk)
        var out = OutputParse(reasoning: split.reasoning)
        if !split.visibleText.isEmpty {
            let parsed = tools.consume(split.visibleText)
            out.visibleText = parsed.visibleText
            out.toolCalls = parsed.toolCalls
        }
        return out
    }

    public mutating func finish() -> OutputParse {
        // Mirrors the engine's old `.info` flush ordering exactly: flush the think parser first, feed
        // its trailing visible text into the tool-call parser, then flush that too.
        let thinkTail = think.finish()
        var out = OutputParse(reasoning: thinkTail.reasoning)
        let tail = tools.consume(thinkTail.visibleText)
        let flushed = tools.finish()
        out.visibleText = tail.visibleText + flushed.visibleText
        out.toolCalls = tail.toolCalls + flushed.toolCalls
        return out
    }
}
