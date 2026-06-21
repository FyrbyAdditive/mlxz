import Testing
@testable import MLXZCore

/// Tests for the Mistral / Llama3 / GLM4 output parsers. Each is fed whole AND char-by-char.
@Suite struct ExtraFormatParserTests {
    private func run(_ parser: () -> any OutputParser, _ input: String, charByChar: Bool) -> OutputParse {
        var p = parser()
        var out = OutputParse()
        if charByChar {
            for ch in input { merge(&out, p.consume(String(ch))) }
        } else {
            merge(&out, p.consume(input))
        }
        merge(&out, p.finish())
        return out
    }
    private func merge(_ out: inout OutputParse, _ r: OutputParse) {
        out.visibleText += r.visibleText; out.reasoning += r.reasoning; out.toolCalls += r.toolCalls
    }

    // MARK: Mistral

    @Test func mistralSingleCall() {
        let input = #"[TOOL_CALLS]get_weather[ARGS]{"location":"Tokyo"}"#
        for cbc in [false, true] {
            let out = run({ MistralOutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.toolCalls.map(\.name) == ["get_weather"])
            #expect(out.toolCalls.first?.argumentsJSON.contains("Tokyo") == true)
            #expect(out.visibleText == "")
        }
    }

    @Test func mistralTextThenCall() {
        let input = #"Sure. [TOOL_CALLS]f[ARGS]{"x":1}"#
        for cbc in [false, true] {
            let out = run({ MistralOutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.visibleText == "Sure. ")
            #expect(out.toolCalls.map(\.name) == ["f"])
        }
    }

    @Test func mistralMultipleCalls() {
        let input = #"[TOOL_CALLS]a[ARGS]{"x":1}[TOOL_CALLS]b[ARGS]{"y":2}"#
        for cbc in [false, true] {
            let out = run({ MistralOutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.toolCalls.map(\.name) == ["a", "b"])
            #expect(Set(out.toolCalls.map(\.id)).count == 2)
        }
    }

    @Test func mistralPlainTextNoLeak() {
        let out = run({ MistralOutputParser(idPrefix: "t") }, "just a normal answer", charByChar: true)
        #expect(out.visibleText == "just a normal answer")
        #expect(out.toolCalls.isEmpty)
    }

    // MARK: Llama3

    @Test func llama3PythonTagCall() {
        let input = #"<|python_tag|>{"name":"f","parameters":{"x":1}}"#
        for cbc in [false, true] {
            let out = run({ Llama3OutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.toolCalls.map(\.name) == ["f"])
            #expect(out.toolCalls.first?.argumentsJSON.contains("\"x\"") == true)
        }
    }

    @Test func llama3InlineJSONCall() {
        let input = #"{"name":"search","arguments":{"q":"hi"}}"#
        for cbc in [false, true] {
            let out = run({ Llama3OutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.toolCalls.map(\.name) == ["search"])
            #expect(out.toolCalls.first?.argumentsJSON.contains("\"q\"") == true)
        }
    }

    @Test func llama3PlainTextPassesThrough() {
        let out = run({ Llama3OutputParser(idPrefix: "t") }, "Hello there!", charByChar: true)
        #expect(out.visibleText == "Hello there!")
        #expect(out.toolCalls.isEmpty)
    }

    @Test func llama3NonCallJSONFlushedAsText() {
        // A JSON object that isn't a tool call (no name) should surface as visible text, not vanish.
        let out = run({ Llama3OutputParser(idPrefix: "t") }, #"{"foo":1}"#, charByChar: false)
        #expect(out.toolCalls.isEmpty)
        #expect(out.visibleText.contains("foo"))
    }

    // MARK: GLM4

    @Test func glm4Call() {
        let input = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
        for cbc in [false, true] {
            let out = run({ GLM4OutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.toolCalls.map(\.name) == ["get_weather"])
            #expect(out.toolCalls.first?.argumentsJSON.contains("city") == true)
            #expect(out.toolCalls.first?.argumentsJSON.contains("Paris") == true)
        }
    }

    @Test func glm4CoercesNumericArg() {
        let input = "<tool_call>f<arg_key>n</arg_key><arg_value>42</arg_value></tool_call>"
        let out = run({ GLM4OutputParser(idPrefix: "t") }, input, charByChar: false)
        // 42 should serialize as a number, not a quoted string.
        #expect(out.toolCalls.first?.argumentsJSON.contains("\"n\":42") == true)
    }

    @Test func glm4TextAroundCall() {
        let input = "ok <tool_call>f<arg_key>a</arg_key><arg_value>b</arg_value></tool_call> done"
        for cbc in [false, true] {
            let out = run({ GLM4OutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.visibleText == "ok  done")
            #expect(out.toolCalls.map(\.name) == ["f"])
        }
    }

    // MARK: Gemma  (real Gemma-4 grammar: <|tool_call>call:name{key:<|"|>v<|"|>}<tool_call|>)

    @Test func gemmaSingleCall() {
        let input = "<|tool_call>call:web_search{query:<|\"|>capital of France<|\"|>}<tool_call|>"
        for cbc in [false, true] {
            let out = run({ GemmaOutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.toolCalls.map(\.name) == ["web_search"])
            #expect(out.toolCalls.first?.argumentsJSON.contains("\"query\"") == true)
            #expect(out.toolCalls.first?.argumentsJSON.contains("capital of France") == true)
            #expect(out.visibleText == "")
        }
    }

    @Test func gemmaMultipleCalls() {
        let input = "<|tool_call>call:web_search{query:<|\"|>a<|\"|>}<tool_call|>"
            + "<|tool_call>call:web_search{query:<|\"|>b<|\"|>}<tool_call|>"
        for cbc in [false, true] {
            let out = run({ GemmaOutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.toolCalls.map(\.name) == ["web_search", "web_search"])
            #expect(Set(out.toolCalls.map(\.id)).count == 2)
        }
    }

    @Test func gemmaStringWithCommaPreserved() {
        // A comma inside the quoted string value must not split the argument.
        let input = "<|tool_call>call:f{q:<|\"|>Paris, France<|\"|>}<tool_call|>"
        let out = run({ GemmaOutputParser(idPrefix: "t") }, input, charByChar: false)
        #expect(out.toolCalls.first?.argumentsJSON.contains("Paris, France") == true)
    }

    @Test func gemmaBareNumericArg() {
        let input = "<|tool_call>call:f{n:42}<tool_call|>"
        let out = run({ GemmaOutputParser(idPrefix: "t") }, input, charByChar: false)
        #expect(out.toolCalls.first?.argumentsJSON.contains("\"n\":42") == true)
    }

    @Test func gemmaTextAroundCall() {
        let input = "Let me look. <|tool_call>call:f{x:<|\"|>y<|\"|>}<tool_call|> done"
        for cbc in [false, true] {
            let out = run({ GemmaOutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.visibleText == "Let me look.  done")
            #expect(out.toolCalls.map(\.name) == ["f"])
        }
    }

    @Test func gemmaPlainTextNoLeak() {
        let out = run({ GemmaOutputParser(idPrefix: "t") }, "Hello! How can I help?", charByChar: true)
        #expect(out.visibleText == "Hello! How can I help?")
        #expect(out.toolCalls.isEmpty)
    }

    @Test func gemmaThoughtRoutesToReasoning() {
        // Real Gemma-4 shape: <|channel>thought\n<channel|> then the channel's content.
        let input = "<|channel>thought\n<channel|>The weather in Paris is 15C and sunny."
        for cbc in [false, true] {
            let out = run({ GemmaOutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.reasoning == "The weather in Paris is 15C and sunny.")
            #expect(out.visibleText == "")
            #expect(!out.reasoning.contains("<|channel"))   // no marker leakage
            #expect(!out.reasoning.contains("<channel|>"))
        }
    }

    @Test func gemmaThoughtMarkersNeverLeakAsVisible() {
        let input = "<|channel>thought <channel|>some reasoning"
        let out = run({ GemmaOutputParser(idPrefix: "t") }, input, charByChar: true)
        #expect(!out.visibleText.contains("<|channel"))
        #expect(!out.visibleText.contains("channel|>"))
        #expect(out.reasoning.contains("some reasoning"))
    }

    @Test func gemmaThoughtThenToolCall() {
        let input = "<|channel>thought\n<channel|>I should search.<|tool_call>call:web_search{q:<|\"|>news<|\"|>}<tool_call|>"
        for cbc in [false, true] {
            let out = run({ GemmaOutputParser(idPrefix: "t") }, input, charByChar: cbc)
            #expect(out.reasoning.contains("I should search."))
            #expect(out.toolCalls.map(\.name) == ["web_search"])
            #expect(out.visibleText == "")
        }
    }

    @Test func gemmaSplitChannelMarkers() {
        var p = GemmaOutputParser(idPrefix: "t")
        var out = OutputParse()
        for chunk in ["<|chan", "nel>thou", "ght<chan", "nel|>cot text"] { merge(&out, p.consume(chunk)) }
        merge(&out, p.finish())
        #expect(out.reasoning == "cot text")
        #expect(out.visibleText == "")
        #expect(!out.reasoning.contains("<|"))
    }
}
