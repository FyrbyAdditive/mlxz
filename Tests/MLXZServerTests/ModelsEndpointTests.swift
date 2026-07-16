import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import MLXZServer
@testable import MLXZCore

@Suite struct ModelsEndpointTests {
    @Test func listsOnlyLoadedModels() async throws {
        let a = MockInferenceEngine(descriptor: .init(repoID: "mock/a"), streamingText: "x")
        let b = MockInferenceEngine(descriptor: .init(repoID: "mock/b"), streamingText: "x")
        let manager = ModelManager(loader: MockModelLoading { d in
            d.repoID == "mock/a" ? a : b
        })
        try await manager.load(a.descriptor)
        try await manager.load(b.descriptor)

        let router = RouterBuilder(
            manager: manager, gate: GenerationGate(), apiKey: nil, logSink: nil
        ).build()
        let app = Application(router: router)

        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/models", method: .get) { response in
                #expect(response.status == .ok)
                let body = String(buffer: response.body)
                #expect(body.contains("\"object\":\"list\""))
                // Strict routing: only LOADED models are listed (both here).
                #expect(body.contains("mock/a"))
                #expect(body.contains("mock/b"))
                // Not a model that was never loaded.
                #expect(!body.contains("mock/never-loaded"))
            }
        }
    }

    @Test func emptyWhenNoModel() async throws {
        let manager = ModelManager(loader: MockModelLoading())
        let router = RouterBuilder(manager: manager, gate: GenerationGate(), apiKey: nil, logSink: nil).build()
        let app = Application(router: router)
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/models", method: .get) { response in
                #expect(response.status == .ok)
                #expect(String(buffer: response.body).contains("\"data\":[]"))
            }
        }
    }
}
