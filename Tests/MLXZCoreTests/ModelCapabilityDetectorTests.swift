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

    @Test func visionConfigSignalDetectsVLMWithoutNameMarker() {
        // Qwen3.6-27B-4bit is a VLM but its repo id has no -vl/vision marker; the config signal
        // (vision_config present / image-text-to-text tag) must still surface vision.
        let withoutSignal = ModelCapabilityDetector.detect(repoID: "mlx-community/Qwen3.6-27B-4bit")
        #expect(!withoutSignal.contains(.vision), "name alone shouldn't flag this as vision")

        let withSignal = ModelCapabilityDetector.detect(
            repoID: "mlx-community/Qwen3.6-27B-4bit", hasVisionConfig: true)
        #expect(withSignal.contains(.vision), "vision-config signal must flag vision")
        #expect(withSignal.contains(.chat))
    }
}
