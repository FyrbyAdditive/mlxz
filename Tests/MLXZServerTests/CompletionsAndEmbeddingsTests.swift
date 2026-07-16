import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import MLXZServer
@testable import MLXZCore

@Suite struct LegacyCompletionsTests {
    private func makeApp(engine: any InferenceEngine) async -> some ApplicationProtocol {
        let manager = ModelManager(loader: MockModelLoading { _ in engine })
        try? await manager.load(engine.descriptor)
        let router = RouterBuilder(manager: manager, gate: GenerationGate(), apiKey: nil, logSink: nil).build()
        return Application(router: router)
    }

    @Test func nonStreamingTextCompletion() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), streamingText: "completed text")
        let app = await makeApp(engine: engine)
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/completions", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"mock/qwen","prompt":"once upon"}"#)
            ) { response in
                #expect(response.status == .ok)
                let body = String(buffer: response.body)
                #expect(body.contains("\"object\":\"text_completion\""))
                #expect(body.contains("completed text"))
                #expect(body.contains("\"finish_reason\":\"stop\""))
            }
        }
    }

    @Test func streamingTextCompletion() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/qwen"), streamingText: "a b")
        let app = await makeApp(engine: engine)
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/completions", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"mock/qwen","stream":true,"prompt":"hi"}"#)
            ) { response in
                #expect(response.status == .ok)
                let body = String(buffer: response.body)
                #expect(body.contains("\"object\":\"text_completion\""))
                #expect(body.contains("data: [DONE]"))
            }
        }
    }
}

@Suite struct EmbeddingsTests {
    private func makeApp() -> some ApplicationProtocol {
        let manager = ModelManager(loader: MockModelLoading())
        let embeddings = EmbeddingManager(loader: MockEmbeddingLoading())
        let router = RouterBuilder(
            manager: manager, gate: GenerationGate(), apiKey: nil, logSink: nil,
            embeddingManager: embeddings
        ).build()
        return Application(router: router)
    }

    @Test func embedsSingleString() async throws {
        let app = makeApp()
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/embeddings", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"mock/embed","input":"hello"}"#)
            ) { response in
                #expect(response.status == .ok)
                let body = String(buffer: response.body)
                #expect(body.contains("\"object\":\"list\""))
                #expect(body.contains("\"object\":\"embedding\""))
                #expect(body.contains("\"embedding\":["))
                #expect(body.contains("\"index\":0"))
            }
        }
    }

    @Test func embedsArrayOfStrings() async throws {
        let app = makeApp()
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/embeddings", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"mock/embed","input":["a","bb","ccc"]}"#)
            ) { response in
                #expect(response.status == .ok)
                let body = String(buffer: response.body)
                // Three embedding objects, indices 0,1,2.
                #expect(body.contains("\"index\":0"))
                #expect(body.contains("\"index\":1"))
                #expect(body.contains("\"index\":2"))
            }
        }
    }

    @Test func missingModelReturns400() async throws {
        let app = makeApp()
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/embeddings", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"input":"hello"}"#)
            ) { response in
                #expect(response.status == .badRequest)
                #expect(String(buffer: response.body).contains("missing_model"))
            }
        }
    }

    @Test func embeddingsNotServedWhenManagerAbsent() async throws {
        let manager = ModelManager(loader: MockModelLoading())
        let router = RouterBuilder(manager: manager, gate: GenerationGate(), apiKey: nil, logSink: nil).build()
        let app = Application(router: router)
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/embeddings", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"m","input":"hi"}"#)
            ) { response in
                // No route registered → 404.
                #expect(response.status == .notFound)
            }
        }
    }
}
