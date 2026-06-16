import Testing
import Foundation
@testable import MLXZHub
@testable import MLXZCore

@Suite struct LocalModelStoreTests {
    /// Build a fake HF cache: <root>/models--org--name/snapshots/<rev>/config.json (+ a weight file).
    private func makeFakeCache(repos: [(repo: String, modelType: String, weightBytes: Int)]) throws -> URL {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("mlxz-test-\(UUID().uuidString)")
        for r in repos {
            let dirName = "models--" + r.repo.replacingOccurrences(of: "/", with: "--")
            let snapshot = root
                .appendingPathComponent(dirName)
                .appendingPathComponent("snapshots")
                .appendingPathComponent("abc123")
            try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
            let config = #"{"model_type":"\#(r.modelType)"}"#
            try config.write(to: snapshot.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
            let weights = Data(count: r.weightBytes)
            try weights.write(to: snapshot.appendingPathComponent("model.safetensors"))
        }
        return root
    }

    @Test func enumeratesInstalledModels() throws {
        let root = try makeFakeCache(repos: [
            ("mlx-community/Qwen3.6-4B-4bit", "qwen3", 1024),
            ("mlx-community/Qwen3.6-35B-A3B-MTP-4bit", "qwen3_5_mtp", 2048),
        ])
        defer { try? FileManager.default.removeItem(at: root) }

        let store = LocalModelStore(cacheRoots: [root])
        let models = store.installedModels().sorted { $0.descriptor.repoID < $1.descriptor.repoID }

        #expect(models.count == 2)
        let mtp = models.first { $0.descriptor.repoID.contains("MTP") }!
        #expect(mtp.modelType == "qwen3_5_mtp")
        #expect(mtp.capabilities.contains(.speculative))
        #expect(mtp.sizeBytes >= 2048)
    }

    @Test func ignoresDirectoriesWithoutConfig() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("mlxz-empty-\(UUID().uuidString)")
        let snapshot = root.appendingPathComponent("models--a--b/snapshots/rev")
        try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        #expect(LocalModelStore(cacheRoots: [root]).installedModels().isEmpty)
    }

    @Test func parsesRepoIDFromCacheDirName() {
        let url = URL(fileURLWithPath: "/tmp/models--mlx-community--Qwen3.6-4B-4bit")
        // No snapshots → nil, but the name parsing is exercised by enumeratesInstalledModels.
        #expect(LocalModelStore.installedModel(at: url) == nil)
    }
}
