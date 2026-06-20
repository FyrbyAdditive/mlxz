import Testing
@testable import MLXZCore

/// Proves `QwenOutputParser` produces the SAME (reasoning, visible, toolCalls) as the original
/// `ThinkParser` + `ToolCallParser` pipeline it wraps — the zero-regression guarantee. Tool-call *ids*
/// differ (random prefix), so we compare names + argumentsJSON, not ids.
@Suite struct QwenOutputParserTests {
    /// Reference pipeline = the exact logic the engine used before `OutputParser`.
    private func reference(_ chunks: [String], startInsideThink: Bool) -> (String, String, [String]) {
        var think = ThinkParser(startInsideThink: startInsideThink)
        var tools = ToolCallParser(idPrefix: "ref")
        var reasoning = "", visible = "", calls: [String] = []
        for c in chunks {
            let s = think.consume(c)
            reasoning += s.reasoning
            if !s.visibleText.isEmpty {
                let p = tools.consume(s.visibleText)
                visible += p.visibleText
                calls += p.toolCalls.map(sig)
            }
        }
        let tt = think.finish()
        reasoning += tt.reasoning
        var tail = tools.consume(tt.visibleText)
        let flushed = tools.finish()
        visible += tail.visibleText + flushed.visibleText
        calls += (tail.toolCalls + flushed.toolCalls).map(sig)
        return (reasoning, visible, calls)
    }

    private func wrapper(_ chunks: [String], startInsideThink: Bool) -> (String, String, [String]) {
        var p = QwenOutputParser(startInsideThink: startInsideThink, idPrefix: "wrap")
        var reasoning = "", visible = "", calls: [String] = []
        for c in chunks {
            let o = p.consume(c)
            reasoning += o.reasoning; visible += o.visibleText; calls += o.toolCalls.map(sig)
        }
        let t = p.finish()
        reasoning += t.reasoning; visible += t.visibleText; calls += t.toolCalls.map(sig)
        return (reasoning, visible, calls)
    }

    private func sig(_ c: ToolCall) -> String { "\(c.name)|\(c.argumentsJSON)" }

    private func assertEquivalent(_ chunks: [String], startInsideThink: Bool = false) {
        let r = reference(chunks, startInsideThink: startInsideThink)
        let w = wrapper(chunks, startInsideThink: startInsideThink)
        #expect(r.0 == w.0, "reasoning differs")
        #expect(r.1 == w.1, "visible differs")
        #expect(r.2 == w.2, "tool calls differ")
    }

    @Test func plainText() { assertEquivalent(["Hello, ", "how can I help?"]) }

    @Test func thinkBlock() {
        assertEquivalent(["<think>reasoning</think>", "visible answer"])
    }

    @Test func preOpenedThink() {
        // Stream starts inside reasoning (template pre-opened <think>); only the close tag appears.
        assertEquivalent(["plan step 1. ", "step 2.</think>", "The answer."], startInsideThink: true)
    }

    @Test func jsonToolCall() {
        assertEquivalent([#"Let me check. <tool_call>{"name":"f","arguments":{"x":1}}</tool_call> done"#])
    }

    @Test func xmlToolCall() {
        assertEquivalent(["<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>"])
    }

    @Test func splitAcrossChunks() {
        assertEquivalent(["<thi", "nk>cot</thi", "nk>vis <tool", "_call>", #"{"name":"x","arguments":{}}"#, "</tool_call>"])
    }
}
