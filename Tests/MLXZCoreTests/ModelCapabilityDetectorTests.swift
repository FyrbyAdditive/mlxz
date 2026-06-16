import Testing
@testable import MLXZCore

@Suite struct ModelCapabilityDetectorTests {
    @Test func plainChatModelHasChatAndTools() {
        let caps = ModelCapabilityDetector.detect(repoID: "mlx-community/Qwen3.6-4B-4bit")
        #expect(caps.contains(.chat))
        #expect(caps.contains(.tools))
        #expect(!caps.contains(.vision))
        #expect(!caps.contains(.speculative))
    }

    @Test func mtpModelIsSpeculative() {
        let caps = ModelCapabilityDetector.detect(repoID: "mlx-community/Qwen3.6-35B-A3B-MTP-4bit")
        #expect(caps.contains(.speculative))
    }

    @Test func mtpDetectedFromModelType() {
        let caps = ModelCapabilityDetector.detect(repoID: "some/model", modelType: "qwen3_5_mtp")
        #expect(caps.contains(.speculative))
    }

    @Test func visionModelDetectedFromName() {
        for repo in ["mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
                     "mlx-community/llava-1.5-7b-4bit",
                     "mlx-community/SmolVLM-Instruct-4bit"] {
            #expect(ModelCapabilityDetector.detect(repoID: repo).contains(.vision), "expected vision for \(repo)")
        }
    }

    @Test func moeChatModelIsStillJustChatTools() {
        // MoE is a serving detail, not a capability flag; Qwen3.6 MoE is a normal chat model.
        let caps = ModelCapabilityDetector.detect(repoID: "mlx-community/Qwen3.6-35B-A3B-4bit")
        #expect(caps.contains(.chat))
        #expect(!caps.contains(.vision))
    }
}
