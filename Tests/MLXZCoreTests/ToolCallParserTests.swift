import Testing
@testable import MLXZCore

@Suite struct ToolCallParserTests {
    /// Feed the whole input as one chunk, then finish.
    private func runWhole(_ input: String) -> ToolCallParser.Output {
        var parser = ToolCallParser()
        var out = parser.consume(input)
        let tail = parser.finish()
        out.visibleText += tail.visibleText
        out.toolCalls += tail.toolCalls
        return out
    }

    /// Feed the input one character at a time — exercises tag-splitting across chunks.
    private func runCharByChar(_ input: String) -> ToolCallParser.Output {
        var parser = ToolCallParser()
        var out = ToolCallParser.Output()
        for ch in input {
            let r = parser.consume(String(ch))
            out.visibleText += r.visibleText
            out.toolCalls += r.toolCalls
        }
        let tail = parser.finish()
        out.visibleText += tail.visibleText
        out.toolCalls += tail.toolCalls
        return out
    }

    @Test func plainTextPassesThrough() {
        let input = "Hello, how can I help you today?"
        #expect(runWhole(input).visibleText == input)
        #expect(runCharByChar(input).visibleText == input)
        #expect(runWhole(input).toolCalls.isEmpty)
    }

    @Test func singleToolCallIsExtracted() {
        let input = #"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.visibleText == "")
            #expect(out.toolCalls.count == 1)
            #expect(out.toolCalls.first?.name == "get_weather")
            // Arguments preserved as JSON object text.
            #expect(out.toolCalls.first?.argumentsJSON.contains("\"city\"") == true)
            #expect(out.toolCalls.first?.argumentsJSON.contains("Paris") == true)
        }
    }

    @Test func textBeforeAndAfterToolCall() {
        let input = #"Let me check. <tool_call>{"name": "f", "arguments": {}}</tool_call> Done."#
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.visibleText == "Let me check.  Done.")
            #expect(out.toolCalls.map(\.name) == ["f"])
        }
    }

    @Test func multipleToolCalls() {
        let input = """
        <tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>
        <tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>
        """
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.toolCalls.map(\.name) == ["a", "b"])
            // Distinct ids.
            #expect(Set(out.toolCalls.map(\.id)).count == 2)
        }
    }

    @Test func openTagSplitAcrossChunks() {
        // Manually split "<tool_call>" mid-tag.
        var parser = ToolCallParser()
        var out = ToolCallParser.Output()
        for chunk in ["Hi <tool", "_call>", #"{"name":"x","arguments":{}}"#, "</tool", "_call> bye"] {
            let r = parser.consume(chunk)
            out.visibleText += r.visibleText
            out.toolCalls += r.toolCalls
        }
        let tail = parser.finish()
        out.visibleText += tail.visibleText
        out.toolCalls += tail.toolCalls

        #expect(out.visibleText == "Hi  bye")
        #expect(out.toolCalls.map(\.name) == ["x"])
        // Crucially, the partial "<tool" was never emitted as visible text.
        #expect(!out.visibleText.contains("tool"))
    }

    @Test func malformedBodyIsDropped() {
        // Not valid JSON inside the tags → no tool call, and the garbage is not leaked as text.
        let input = "<tool_call>not json</tool_call>"
        let out = runWhole(input)
        #expect(out.toolCalls.isEmpty)
        #expect(out.visibleText == "")
    }

    @Test func argumentsAsStringRoundTrip() {
        // Some templates emit arguments as a pre-serialized string.
        let input = #"<tool_call>{"name": "f", "arguments": "{\"k\": \"v\"}"}</tool_call>"#
        let out = runWhole(input)
        #expect(out.toolCalls.first?.name == "f")
        #expect(out.toolCalls.first?.argumentsJSON.contains("\"k\"") == true)
    }

    @Test func lessThanSignThatIsNotATagPassesThrough() {
        let input = "if a < b and c > d then"
        #expect(runWhole(input).visibleText == input)
        #expect(runCharByChar(input).visibleText == input)
    }
}
