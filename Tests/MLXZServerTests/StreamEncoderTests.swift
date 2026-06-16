import Testing
@testable import MLXZServer
@testable import MLXZCore

@Suite struct StreamEncoderTests {
    @Test func textDeltaSendsRoleOnceThenContent() throws {
        let enc = ChatCompletionStreamEncoder(modelID: "qwen")

        let first = try enc.encode(.textDelta("Hello"))
        #expect(first.count == 1)
        #expect(first[0].event == nil)
        #expect(first[0].data.contains("\"role\":\"assistant\""))
        #expect(first[0].data.contains("\"content\":\"Hello\""))
        #expect(first[0].data.contains("\"object\":\"chat.completion.chunk\""))

        let second = try enc.encode(.textDelta(" world"))
        // Role must NOT repeat after the first delta.
        #expect(!second[0].data.contains("\"role\""))
        #expect(second[0].data.contains("\"content\":\" world\""))
    }

    @Test func completedEmitsFinishReason() throws {
        let enc = ChatCompletionStreamEncoder(modelID: "qwen")
        let frames = try enc.encode(.completed(.init(finishReason: .stop)))
        #expect(frames[0].data.contains("\"finish_reason\":\"stop\""))
    }

    @Test func toolCallChunkShape() throws {
        let enc = ChatCompletionStreamEncoder(modelID: "qwen")
        let call = ToolCall(id: "call_1", name: "f", argumentsJSON: #"{"x":1}"#)
        let frames = try enc.encode(.toolCall(call))
        #expect(frames[0].data.contains("\"tool_calls\""))
        #expect(frames[0].data.contains("\"name\":\"f\""))
        #expect(frames[0].data.contains("\"index\":0"))
    }

    @Test func terminatorIsDone() {
        let enc = ChatCompletionStreamEncoder(modelID: "qwen")
        let term = enc.terminator()
        #expect(term == [SSEFrame(event: nil, data: "[DONE]")])
        #expect(term[0].wireText == "data: [DONE]\n\n")
    }

    @Test func startedProducesNoFrames() throws {
        let enc = ChatCompletionStreamEncoder(modelID: "qwen")
        #expect(try enc.encode(.started(.init(modelID: "qwen"))).isEmpty)
    }
}
