import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import MLXZServer
@testable import MLXZCore

@Suite struct ModelsEndpointTests {
    @Test func listsLoadedAndExtraModels() async throws {
        let engine = MockInferenceEngine(descriptor: .init(repoID: "mock/loaded"), streamingText: "x")
        let manager = ModelManager(loader: MockModelLoading { _ in engine })
        try await manager.load(engine.descriptor)

        let router = RouterBuilder(
            manager: manager,
            gate: GenerationGate(),
            apiKey: nil,
            logSink: nil,
            extraModelIDs: { ["mock/installed-a", "mock/installed-b", "mock/loaded"] }
        ).build()
        let app = Application(router: router)

        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/models", method: .get) { response in
                #expect(response.status == .ok)
                let body = String(buffer: response.body)
                #expect(body.contains("\"object\":\"list\""))
                #expect(body.contains("mock/loaded"))
                #expect(body.contains("mock/installed-a"))
                // De-duplicated: "mock/loaded" appears once.
                let occurrences = body.components(separatedBy: "mock/loaded").count - 1
                #expect(occurrences == 1)
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
