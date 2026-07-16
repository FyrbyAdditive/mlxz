import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import MLXZServer
@testable import MLXZCore

@Suite struct ChatCompletionsTests {
    /// Build a test app with a mock engine loaded into the manager.
    private func makeApp(
        engine: any InferenceEngine,
        apiKey: String? = nil
    ) async -> some ApplicationProtocol {
        let manager = ModelManager(loader: MockModelLoading { _ in engine })
        try? await manager.load(engine.descriptor)
        let router = RouterBuilder(
            manager: manager,
            gate: GenerationGate(maxConcurrent: 1),
            apiKey: apiKey,
            logSink: nil
        ).build()
        return Application(router: router)
    }

    private func chatBody(stream: Bool, content: String = "hi") -> ByteBuffer {
        let json = """
        {"model":"mock/qwen","stream":\(stream),"messages":[{"role":"user","content":"\(content)"}]}
        """
        return ByteBuffer(string: json)
    }

    @Test func nonStreamingReturnsAssistantMessage() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), streamingText: "Hello there")
        let app = await makeApp(engine: engine)

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: chatBody(stream: false)
            ) { response in
                #expect(response.status == .ok)
                let body = String(buffer: response.body)
                #expect(body.contains("\"object\":\"chat.completion\""))
                #expect(body.contains("Hello there"))
                #expect(body.contains("\"role\":\"assistant\""))
                #expect(body.contains("\"finish_reason\":\"stop\""))
            }
        }
    }

    @Test func streamingEmitsChunksAndDone() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), streamingText: "one two")
        let app = await makeApp(engine: engine)

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: chatBody(stream: true)
            ) { response in
                #expect(response.status == .ok)
                let ct = response.headers[.contentType]
                #expect(ct == "text/event-stream")
                let body = String(buffer: response.body)
                #expect(body.contains("\"object\":\"chat.completion.chunk\""))
                #expect(body.contains("data: [DONE]"))
                // The role appears once, in the first delta.
                #expect(body.contains("\"role\":\"assistant\""))
            }
        }
    }

    @Test func toolCallSurfacesInNonStreamingResponse() async throws {
        // Engine emits a tool call event directly.
        let call = ToolCall(id: "call_1", name: "get_weather", argumentsJSON: #"{"city":"Paris"}"#)
        let events: [GenerationEvent] = [
            .started(.init(modelID: "mock/qwen")),
            .toolCall(call),
            .completed(.init(finishReason: .toolCalls, usage: .init(promptTokens: 5, completionTokens: 3))),
        ]
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), capabilities: [.chat, .tools], events: events)
        let app = await makeApp(engine: engine)

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: chatBody(stream: false)
            ) { response in
                #expect(response.status == .ok)
                let body = String(buffer: response.body)
                #expect(body.contains("\"tool_calls\""))
                #expect(body.contains("get_weather"))
                #expect(body.contains("\"finish_reason\":\"tool_calls\""))
            }
        }
    }

    @Test func noModelLoadedReturnsOpenAIError() async throws {
        let manager = ModelManager(loader: MockModelLoading())
        // Deliberately do NOT load a model.
        let router = RouterBuilder(manager: manager, gate: GenerationGate(), apiKey: nil, logSink: nil).build()
        let app = Application(router: router)

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: chatBody(stream: false)
            ) { response in
                #expect(response.status == .notFound)
                let body = String(buffer: response.body)
                #expect(body.contains("\"error\""))
                // Strict routing: a named-but-unloaded model → model_not_found.
                #expect(body.contains("model_not_found"))
            }
        }
    }

    @Test func requestForUnloadedModelIs404() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), streamingText: "x")
        let manager = ModelManager(loader: MockModelLoading { _ in engine })
        try await manager.load(engine.descriptor)   // mock/qwen loaded
        let router = RouterBuilder(manager: manager, gate: GenerationGate(), apiKey: nil, logSink: nil).build()
        let app = Application(router: router)
        try await app.test(.router) { client in
            // Request a DIFFERENT, not-loaded model → 404 even though one model is loaded.
            let body = ByteBuffer(string: #"{"model":"other/model","messages":[{"role":"user","content":"hi"}]}"#)
            try await client.execute(
                uri: "/v1/chat/completions", method: .post,
                headers: [.contentType: "application/json"], body: body
            ) { response in
                #expect(response.status == .notFound)
                #expect(String(buffer: response.body).contains("model_not_found"))
            }
        }
    }

    @Test func malformedBodyReturns400() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), streamingText: "x")
        let app = await makeApp(engine: engine)

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: "{not json")
            ) { response in
                #expect(response.status == .badRequest)
                #expect(String(buffer: response.body).contains("invalid_body"))
            }
        }
    }

    @Test func apiKeyEnforcedWhenConfigured() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), streamingText: "x")
        let app = await makeApp(engine: engine, apiKey: "secret")

        try await app.test(.router) { client in
            // Missing key → 401.
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: chatBody(stream: false)
            ) { response in
                #expect(response.status == .unauthorized)
            }
            // Correct key → 200.
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json", .authorization: "Bearer secret"],
                body: chatBody(stream: false)
            ) { response in
                #expect(response.status == .ok)
            }
        }
    }

    @Test func healthCheckOK() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), streamingText: "x")
        let app = await makeApp(engine: engine)
        try await app.test(.router) { client in
            try await client.execute(uri: "/health", method: .get) { response in
                #expect(response.status == .ok)
                #expect(String(buffer: response.body).contains("ok"))
            }
        }
    }
}
