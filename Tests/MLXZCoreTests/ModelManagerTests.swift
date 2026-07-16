import Testing
@testable import MLXZCore

@Suite struct ModelManagerTests {
    @Test func loadsModelAndExposesEngine() async throws {
        let loader = MockModelLoading()
        let manager = ModelManager(loader: loader)

        #expect(await manager.currentEngine() == nil)

        let descriptor = ModelDescriptor(repoID: "mlx-community/Qwen3.6-4B-4bit")
        try await manager.load(descriptor)

        // Exactly one loaded → currentEngine() returns it; engine(for:) resolves by id.
        #expect(await manager.currentEngine() != nil)
        #expect(await manager.engine(for: descriptor.repoID) != nil)
        #expect(await manager.snapshot.loadedDescriptor == descriptor)
        #expect(await loader.requestedDescriptors == [descriptor])
    }

    @Test func attachesDrafterAndTracksIt() async throws {
        let loader = MockModelLoading()
        let manager = ModelManager(loader: loader)
        let base = ModelDescriptor(repoID: "mlx-community/Qwen3.6-27B-4bit")
        let drafter = "mlx-community/Qwen3.6-27B-MTP-4bit"

        try await manager.load(base, draftModelID: drafter)

        #expect(await manager.snapshot.loadedDescriptor == base)
        #expect(await manager.snapshot.attachedDrafterID == drafter)
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

    // MARK: - Multiple models

    @Test func loadsMultipleModelsSimultaneously() async throws {
        let manager = ModelManager(loader: MockModelLoading())
        let a = ModelDescriptor(repoID: "org/a")
        let b = ModelDescriptor(repoID: "org/b")

        try await manager.load(a)
        try await manager.load(b)

        // Both resident (load does NOT unload the previous).
        #expect(await manager.snapshot.loaded.count == 2)
        #expect(await manager.engine(for: "org/a") != nil)
        #expect(await manager.engine(for: "org/b") != nil)
        // With >1 loaded, currentEngine() is ambiguous → nil.
        #expect(await manager.currentEngine() == nil)
        // Newest first.
        #expect(await manager.loadedModelIDs() == ["org/b", "org/a"])
    }

    @Test func engineForUnknownIDIsNil() async throws {
        let manager = ModelManager(loader: MockModelLoading())
        try await manager.load(ModelDescriptor(repoID: "org/a"))
        #expect(await manager.engine(for: "org/missing") == nil)
    }

    @Test func unloadOneKeepsOthers() async throws {
        let manager = ModelManager(loader: MockModelLoading())
        try await manager.load(ModelDescriptor(repoID: "org/a"))
        try await manager.load(ModelDescriptor(repoID: "org/b"))

        await manager.unload("org/a")

        #expect(await manager.engine(for: "org/a") == nil)
        #expect(await manager.engine(for: "org/b") != nil)
        #expect(await manager.snapshot.loaded.count == 1)
    }

    @Test func evictLeastRecentlyUsedDropsOldest() async throws {
        let manager = ModelManager(loader: MockModelLoading())
        try await manager.load(ModelDescriptor(repoID: "org/a"))
        try await manager.load(ModelDescriptor(repoID: "org/b"))
        // Touch a so b becomes the least-recently-used.
        _ = await manager.engine(for: "org/a")

        let evicted = await manager.evictLeastRecentlyUsed()
        #expect(evicted == "org/b")
        #expect(await manager.engine(for: "org/b") == nil)
        #expect(await manager.engine(for: "org/a") != nil)
    }

    @Test func unloadAllClearsEverything() async throws {
        let manager = ModelManager(loader: MockModelLoading())
        try await manager.load(ModelDescriptor(repoID: "org/a"))
        try await manager.load(ModelDescriptor(repoID: "org/b"))
        await manager.unloadAll()
        #expect(await manager.snapshot.loaded.isEmpty)
        #expect(await manager.currentEngine() == nil)
    }

    @Test func failedLoadSurfacesFailureState() async {
        let manager = ModelManager(loader: ThrowingLoader())
        await #expect(throws: (any Error).self) {
            try await manager.load(ModelDescriptor(repoID: "bad/model"))
        }
        #expect(await manager.snapshot.failed.contains { $0.descriptor.repoID == "bad/model" })
        #expect(await manager.snapshot.loaded.isEmpty)
    }

    @Test func observersReceiveCurrentStateImmediately() async throws {
        let manager = ModelManager(loader: MockModelLoading())
        var iterator = await manager.states().makeAsyncIterator()
        let first = await iterator.next()
        #expect(first?.loaded.isEmpty == true)
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
