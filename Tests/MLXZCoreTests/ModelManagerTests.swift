import Testing
@testable import MLXZCore

@Suite struct ModelManagerTests {
    @Test func loadsModelAndExposesEngine() async throws {
        let loader = MockModelLoading()
        let manager = ModelManager(loader: loader)

        #expect(await manager.currentEngine() == nil)

        let descriptor = ModelDescriptor(repoID: "mlx-community/Qwen3.6-4B-4bit")
        try await manager.load(descriptor)

        let engine = await manager.currentEngine()
        #expect(engine != nil)
        #expect(await manager.state.loadedDescriptor == descriptor)
        #expect(await loader.requestedDescriptors == [descriptor])
    }

    @Test func attachesDrafterAndTracksIt() async throws {
        let loader = MockModelLoading()
        let manager = ModelManager(loader: loader)
        let base = ModelDescriptor(repoID: "mlx-community/Qwen3.6-27B-4bit")
        let drafter = "mlx-community/Qwen3.6-27B-MTP-4bit"

        try await manager.load(base, draftModelID: drafter)

        #expect(await manager.state.loadedDescriptor == base)
        #expect(await manager.state.attachedDrafterID == drafter)
        #expect(await loader.requestedDrafters == [drafter])
    }

    @Test func loadingSameModelTwiceIsIdempotent() async throws {
        let loader = MockModelLoading()
        let manager = ModelManager(loader: loader)
        let descriptor = ModelDescriptor(repoID: "a/b")

        try await manager.load(descriptor)
        try await manager.load(descriptor)

        // Second load should be a no-op (not re-requested from the loader).
        #expect(await loader.requestedDescriptors == [descriptor])
    }

    @Test func unloadClearsEngine() async throws {
        let manager = ModelManager(loader: MockModelLoading())
        try await manager.load(ModelDescriptor(repoID: "a/b"))
        await manager.unload()
        #expect(await manager.currentEngine() == nil)
        if case .empty = await manager.state {} else {
            Issue.record("expected .empty state after unload")
        }
    }

    @Test func failedLoadSurfacesFailureState() async {
        struct LoadFailure: Error {}
        let loader = MockModelLoading { _ in
            // unreachable; we throw before returning
            MockInferenceEngine(streamingText: "")
        }
        // Wrap a loader that always throws.
        let throwing = ThrowingLoader()
        let manager = ModelManager(loader: throwing)
        _ = loader

        await #expect(throws: (any Error).self) {
            try await manager.load(ModelDescriptor(repoID: "bad/model"))
        }
        if case .failed = await manager.state {} else {
            Issue.record("expected .failed state")
        }
    }

    @Test func observersReceiveCurrentStateImmediately() async throws {
        let manager = ModelManager(loader: MockModelLoading())
        var iterator = await manager.states().makeAsyncIterator()
        let first = await iterator.next()
        if case .empty = first {} else {
            Issue.record("expected initial .empty state on subscription")
        }
    }
}

private struct ThrowingLoader: ModelLoading {
    struct Boom: Error {}
    func load(
        _ descriptor: ModelDescriptor,
        draftModelID: String?,
        progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws -> any InferenceEngine {
        throw Boom()
    }
}
