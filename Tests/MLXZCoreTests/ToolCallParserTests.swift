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

    // MARK: Qwen3.5/3.6 XML tool-call format
    //   <tool_call><function=NAME><parameter=ARG>value</parameter>...</function></tool_call>
    // This is what the Qwen3.6 chat template instructs the model to emit — NOT JSON. Without
    // parsing it, the call is silently dropped (empty content, no tool call) and the agent stalls.

    @Test func qwenXMLToolCallSingleParam() {
        let input = """
        <tool_call>
        <function=list_dir>
        <parameter=path>
        /tmp
        </parameter>
        </function>
        </tool_call>
        """
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.toolCalls.count == 1)
            #expect(out.toolCalls.first?.name == "list_dir")
            #expect(out.toolCalls.first?.argumentsJSON.contains("\"path\"") == true)
            #expect(out.toolCalls.first?.argumentsJSON.contains("/tmp") == true)
            #expect(out.visibleText.trimmingCharacters(in: .whitespacesAndNewlines) == "")
        }
    }

    @Test func qwenXMLToolCallMultipleParams() {
        let input = """
        <tool_call>
        <function=replace_string>
        <parameter=path>
        a.txt
        </parameter>
        <parameter=count>
        3
        </parameter>
        </function>
        </tool_call>
        """
        let out = runWhole(input)
        #expect(out.toolCalls.first?.name == "replace_string")
        let args = out.toolCalls.first?.argumentsJSON ?? ""
        #expect(args.contains("\"path\"") && args.contains("a.txt"))
        #expect(args.contains("\"count\""))
    }

    @Test func qwenXMLToolCallNoParams() {
        let input = "<tool_call>\n<function=get_changed_files>\n</function>\n</tool_call>"
        let out = runWhole(input)
        #expect(out.toolCalls.first?.name == "get_changed_files")
        #expect(out.toolCalls.first?.argumentsJSON == "{}")
    }

    @Test func toolCallIdsAreUniqueAcrossParsers() {
        // Each request builds a fresh parser. Ids must NOT collide across requests/turns, or
        // Copilot's call→result matching breaks and the agent stalls (re-issues calls, empties out).
        func firstID(_ input: String) -> String? {
            var p = ToolCallParser()
            var out = p.consume(input)
            out.toolCalls += p.finish().toolCalls
            return out.toolCalls.first?.id
        }
        let a = firstID(#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#)
        let b = firstID(#"<tool_call>{"name":"g","arguments":{}}</tool_call>"#)
        #expect(a != nil && b != nil)
        #expect(a != b, "tool-call ids must be unique across separate parsers (turns)")
    }

    @Test func qwenXMLToolCallWithSurroundingText() {
        let input = "Sure. <tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call> ok"
        let out = runWhole(input)
        #expect(out.toolCalls.map(\.name) == ["f"])
        #expect(out.visibleText.contains("Sure."))
        #expect(out.visibleText.contains("ok"))
    }

    @Test func lessThanSignThatIsNotATagPassesThrough() {
        let input = "if a < b and c > d then"
        #expect(runWhole(input).visibleText == input)
        #expect(runCharByChar(input).visibleText == input)
    }
}
