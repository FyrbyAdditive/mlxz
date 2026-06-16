import Testing
import Foundation
@testable import MLXZInference
@testable import MLXZCore

/// Integration tests that download and run a real (small) model. Gated behind an env flag
/// because they are slow, network-bound, and require Apple Silicon + the Metal toolchain.
/// Run with: `MLXZ_INTEGRATION=1 xcodebuild test -scheme mlxz ...`
@Suite(.enabled(if: ProcessInfo.processInfo.environment["MLXZ_INTEGRATION"] == "1"))
struct EngineIntegrationTests {
    /// A small model keeps the download manageable.
    static let testModel = ModelDescriptor(repoID: "mlx-community/Qwen3.6-0.6B-4bit")

    @Test func loadsAndGeneratesText() async throws {
        let loader = MLXModelLoader()
        let engine = try await loader.load(Self.testModel) { _ in }

        #expect(engine.capabilities.contains(.chat))

        let request = GenerationRequest(
            messages: [ChatMessage(role: .user, text: "Reply with exactly: pong")],
            maxTokens: 16
        )
        var text = ""
        var completed = false
        for try await event in try await engine.generate(request) {
            switch event {
            case .textDelta(let t): text += t
            case .completed: completed = true
            default: break
            }
        }
        #expect(completed)
        #expect(!text.isEmpty)
    }
}
