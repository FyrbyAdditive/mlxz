import Testing
@testable import MLXZCore

@Suite struct HarmonyOutputParserTests {
    /// Feed the whole input as one chunk, then finish.
    private func runWhole(_ input: String) -> OutputParse {
        var parser = HarmonyOutputParser(idPrefix: "test")
        var out = parser.consume(input)
        let tail = parser.finish()
        merge(&out, tail)
        return out
    }

    /// Feed one character at a time — exercises marker-splitting across chunks.
    private func runCharByChar(_ input: String) -> OutputParse {
        var parser = HarmonyOutputParser(idPrefix: "test")
        var out = OutputParse()
        for ch in input { merge(&out, parser.consume(String(ch))) }
        merge(&out, parser.finish())
        return out
    }

    private func merge(_ out: inout OutputParse, _ r: OutputParse) {
        out.visibleText += r.visibleText
        out.reasoning += r.reasoning
        out.toolCalls += r.toolCalls
    }

    @Test func analysisRoutesToReasoning() {
        let input = "<|channel|>analysis<|message|>We should think.<|end|>"
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.reasoning == "We should think.")
            #expect(out.visibleText == "")
            #expect(out.toolCalls.isEmpty)
        }
    }

    @Test func finalRoutesToVisible() {
        let input = "<|channel|>final<|message|>The sky is blue.<|return|>"
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.visibleText == "The sky is blue.")
            #expect(out.reasoning == "")
            #expect(out.toolCalls.isEmpty)
        }
    }

    @Test func singleToolCall() {
        let input = #"<|channel|>commentary to=functions.web_search <|constrain|>json<|message|>{"query":"news","max_results":10}<|call|>"#
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.toolCalls.count == 1)
            #expect(out.toolCalls.first?.name == "web_search")
            #expect(out.toolCalls.first?.argumentsJSON.contains("\"query\"") == true)
            #expect(out.toolCalls.first?.argumentsJSON.contains("\"max_results\"") == true)
            #expect(out.visibleText == "")
        }
    }

    @Test func toolNameOnStartMarker() {
        // The `to=functions.NAME` can appear on the <|start|> role marker instead of the channel.
        let input = #"<|start|>assistant to=functions.get_weather<|channel|>commentary<|message|>{"city":"Paris"}<|call|>"#
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.toolCalls.map(\.name) == ["get_weather"])
            #expect(out.toolCalls.first?.argumentsJSON.contains("Paris") == true)
        }
    }

    @Test func multipleToolCalls() {
        let input = """
        <|channel|>commentary to=functions.a <|constrain|>json<|message|>{"x":1}<|call|>\
        <|start|>assistant<|channel|>commentary to=functions.b <|constrain|>json<|message|>{"y":2}<|call|>
        """
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.toolCalls.map(\.name) == ["a", "b"])
            #expect(Set(out.toolCalls.map(\.id)).count == 2)   // distinct ids
        }
    }

    @Test func smartQuotesNormalized() {
        // gpt-oss sometimes emits curly quotes in args — must still parse (the web_search regression).
        let input = "<|channel|>commentary to=functions.web_search <|constrain|>json<|message|>"
            + "{\u{201C}query\u{201D}:\u{201C}today\u{2019}s news\u{201D}}<|call|>"
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.toolCalls.count == 1)
            #expect(out.toolCalls.first?.name == "web_search")
            #expect(out.toolCalls.first?.argumentsJSON.contains("query") == true)
        }
    }

    @Test func interleavedAnalysisThenFinal() {
        let input = "<|channel|>analysis<|message|>Reasoning here.<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Answer here.<|return|>"
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.reasoning == "Reasoning here.")
            #expect(out.visibleText == "Answer here.")
            #expect(out.toolCalls.isEmpty)
        }
    }

    @Test func splitMarkersAcrossChunks() {
        // Feed markers broken at awkward points; no fragment must leak into reasoning/visible.
        var parser = HarmonyOutputParser(idPrefix: "test")
        var out = OutputParse()
        for chunk in ["<|chan", "nel|>analy", "sis<|mess", "age|>cot text<|re", "turn|>"] {
            merge(&out, parser.consume(chunk))
        }
        merge(&out, parser.finish())
        #expect(out.reasoning == "cot text")
        #expect(out.visibleText == "")
        #expect(!out.reasoning.contains("<|"))
        #expect(!out.visibleText.contains("<|"))
    }

    @Test func missingEndDefensiveFlush() {
        // An analysis block with no terminal then EOS → finish() must still emit the buffered reasoning.
        let input = "<|channel|>analysis<|message|>unterminated reasoning"
        for out in [runWhole(input), runCharByChar(input)] {
            #expect(out.reasoning == "unterminated reasoning")
        }
    }

    @Test func returnEndsBlock() {
        // <|return|> (EOS) terminates a final block even without <|end|>.
        let out = runWhole("<|channel|>final<|message|>done<|return|>trailing ignored")
        #expect(out.visibleText == "done")
    }

    @Test func idsUniqueAcrossParsers() {
        // Fresh parsers (per request) must not collide — clients match results to calls by id.
        let input = #"<|channel|>commentary to=functions.f <|constrain|>json<|message|>{}<|call|>"#
        var p1 = HarmonyOutputParser(); var p2 = HarmonyOutputParser()
        var o1 = p1.consume(input); o1.toolCalls += p1.finish().toolCalls
        var o2 = p2.consume(input); o2.toolCalls += p2.finish().toolCalls
        #expect(o1.toolCalls.first?.id != o2.toolCalls.first?.id)
    }

    @Test func emptyArgsBecomeEmptyObject() {
        let out = runWhole(#"<|channel|>commentary to=functions.ping <|constrain|>json<|message|>{}<|call|>"#)
        #expect(out.toolCalls.map(\.name) == ["ping"])
        #expect(out.toolCalls.first?.argumentsJSON == "{}")
    }
}
