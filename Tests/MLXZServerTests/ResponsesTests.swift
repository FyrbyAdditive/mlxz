import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import MLXZServer
@testable import MLXZCore

@Suite struct ResponsesEncoderTests {
    /// Collect all events for a simple text generation.
    private func runText(_ text: String) throws -> [SSEFrame] {
        let enc = ResponsesStreamEncoder(modelID: "qwen")
        var frames: [SSEFrame] = []
        frames += try enc.encode(.started(.init(modelID: "qwen")))
        for word in text.split(separator: " ") {
            frames += try enc.encode(.textDelta(String(word)))
        }
        frames += try enc.encode(.completed(.init(finishReason: .stop, usage: .init(promptTokens: 3, completionTokens: 2))))
        frames += enc.terminator()
        return frames
    }

    @Test func emitsCanonicalEventSequence() throws {
        let frames = try runText("hello world")
        let events = frames.map(\.event)
        #expect(events.first == "response.created")
        #expect(events.contains("response.output_item.added"))
        #expect(events.contains("response.content_part.added"))
        #expect(events.contains("response.output_text.delta"))
        #expect(events.contains("response.output_text.done"))
        #expect(events.contains("response.content_part.done"))
        #expect(events.contains("response.output_item.done"))
        #expect(events.last == "response.completed")
    }

    @Test func dataCarriesTypeAndSequenceNumber() throws {
        let frames = try runText("hi")
        // Every frame's data JSON must include "type" and "sequence_number".
        for f in frames {
            #expect(f.data.contains("\"type\""))
            #expect(f.data.contains("\"sequence_number\""))
        }
        // sequence_numbers are strictly increasing 0,1,2,...
        let seqs = frames.compactMap { f -> Int? in
            guard let r = f.data.range(of: "\"sequence_number\":") else { return nil }
            let rest = f.data[r.upperBound...].prefix(while: { $0.isNumber })
            return Int(rest)
        }
        #expect(seqs == Array(0..<seqs.count))
    }

    @Test func noDoneTerminator() throws {
        // Unlike chat completions, the Responses stream has no data:[DONE].
        let frames = try runText("hi")
        #expect(!frames.contains { $0.data == "[DONE]" })
    }

    @Test func completedCarriesOutputTextAndUsage() throws {
        let frames = try runText("hello world")
        let completed = frames.last!
        #expect(completed.event == "response.completed")
        #expect(completed.data.contains("\"output_text\""))
        #expect(completed.data.contains("helloworld") || completed.data.contains("hello"))
        #expect(completed.data.contains("\"output_tokens\":2"))
        #expect(completed.data.contains("\"status\":\"completed\""))
    }

    @Test func toolCallEmitsFunctionCallItem() throws {
        let enc = ResponsesStreamEncoder(modelID: "qwen")
        var frames: [SSEFrame] = []
        frames += try enc.encode(.started(.init(modelID: "qwen")))
        frames += try enc.encode(.toolCall(ToolCall(id: "call_1", name: "get_weather", argumentsJSON: #"{"city":"Paris"}"#)))
        frames += try enc.encode(.completed(.init(finishReason: .toolCalls)))
        let events = frames.map(\.event)
        #expect(events.contains("response.function_call_arguments.delta"))
        #expect(events.contains("response.function_call_arguments.done"))
        #expect(frames.contains { $0.data.contains("get_weather") })
        #expect(frames.contains { $0.data.contains("\"function_call\"") })
    }
}

@Suite struct ResponsesHTTPTests {
    private func makeApp(engine: any InferenceEngine) async -> some ApplicationProtocol {
        let manager = ModelManager(loader: MockModelLoading { _ in engine })
        try? await manager.load(engine.descriptor)
        let router = RouterBuilder(manager: manager, gate: GenerationGate(), apiKey: nil, logSink: nil).build()
        return Application(router: router)
    }

    @Test func nonStreamingResponse() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), streamingText: "Hello there")
        let app = await makeApp(engine: engine)
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/responses",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"qwen","input":"hi"}"#)
            ) { response in
                #expect(response.status == .ok)
                let body = String(buffer: response.body)
                #expect(body.contains("\"object\":\"response\""))
                #expect(body.contains("\"output_text\""))
                #expect(body.contains("Hello there"))
                #expect(body.contains("\"status\":\"completed\""))
            }
        }
    }

    @Test func streamingResponseEmitsStructuredEvents() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), streamingText: "one two")
        let app = await makeApp(engine: engine)
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/responses",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"qwen","stream":true,"input":"hi"}"#)
            ) { response in
                #expect(response.status == .ok)
                #expect(response.headers[.contentType] == "text/event-stream")
                let body = String(buffer: response.body)
                #expect(body.contains("event: response.created"))
                #expect(body.contains("event: response.output_text.delta"))
                #expect(body.contains("event: response.completed"))
                #expect(!body.contains("[DONE]"))
            }
        }
    }

    @Test func acceptsStructuredInputItems() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), streamingText: "ok")
        let app = await makeApp(engine: engine)
        let body = #"{"model":"qwen","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}]}"#
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/responses", method: .post,
                headers: [.contentType: "application/json"], body: ByteBuffer(string: body)
            ) { response in
                #expect(response.status == .ok)
            }
        }
    }
}
