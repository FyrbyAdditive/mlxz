import Foundation
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

    @Test func noUsageChunkWhenIncludeUsageNotSet() throws {
        // Default (include_usage absent): no trailing usage chunk — just the finish chunk.
        let enc = ChatCompletionStreamEncoder(modelID: "qwen")
        let frames = try enc.encode(
            .completed(.init(finishReason: .stop, usage: .init(promptTokens: 10, completionTokens: 5))))
        #expect(frames.count == 1)
        #expect(!frames.contains { $0.data.contains("\"usage\"") })
    }

    @Test func includeUsageEmitsTrailingUsageChunk() throws {
        // With include_usage: a second chunk with empty `choices` and full `usage` (all three
        // fields) — what VS Code Copilot needs to drive the context-window % counter.
        let enc = ChatCompletionStreamEncoder(modelID: "qwen", includeUsage: true)
        let frames = try enc.encode(
            .completed(.init(finishReason: .stop, usage: .init(promptTokens: 1234, completionTokens: 56))))
        #expect(frames.count == 2)
        // First frame: the normal finish chunk (no usage).
        #expect(frames[0].data.contains("\"finish_reason\":\"stop\""))
        // Second frame: empty choices + usage with all three token fields.
        let usage = frames[1].data
        #expect(usage.contains("\"choices\":[]"))
        #expect(usage.contains("\"prompt_tokens\":1234"))
        #expect(usage.contains("\"completion_tokens\":56"))
        #expect(usage.contains("\"total_tokens\":1290"))
    }

    @Test func includeUsageParsedFromWire() throws {
        // stream_options.include_usage decodes and drives the encoder.
        let json = #"""
        {"model":"q","stream":true,"stream_options":{"include_usage":true},
         "messages":[{"role":"user","content":"hi"}]}
        """#
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        #expect(req.includeUsage == true)

        let without = #"{"model":"q","stream":true,"messages":[{"role":"user","content":"hi"}]}"#
        let req2 = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(without.utf8))
        #expect(req2.includeUsage == false)
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
