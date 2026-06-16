import Testing
import Foundation
@testable import MLXZHub
@testable import MLXZCore

@Suite struct CopilotConfigTests {
    @Test func generatesValidJSONEntry() throws {
        let entry = CopilotConfig.modelEntry(
            repoID: "mlx-community/Qwen3.6-4B-4bit",
            host: "127.0.0.1",
            port: 8080,
            capabilities: [.chat, .tools],
            maxInputTokens: 128_000
        )
        // Must parse as JSON.
        let data = Data(entry.utf8)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(obj?["id"] as? String == "mlx-community/Qwen3.6-4B-4bit")
        #expect(obj?["url"] as? String == "http://127.0.0.1:8080/v1")
        #expect(obj?["apiType"] as? String == "chat-completions")
        #expect(obj?["toolCalling"] as? Bool == true)
        #expect(obj?["vision"] as? Bool == false)
        #expect(obj?["maxInputTokens"] as? Int == 128_000)
    }

    @Test func visionFlagReflectsCapabilities() throws {
        let entry = CopilotConfig.modelEntry(
            repoID: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            capabilities: [.chat, .tools, .vision]
        )
        let obj = try JSONSerialization.jsonObject(with: Data(entry.utf8)) as? [String: Any]
        #expect(obj?["vision"] as? Bool == true)
    }

    @Test func responsesApiTypeHonored() throws {
        let entry = CopilotConfig.modelEntry(
            repoID: "a/b", capabilities: [.chat], apiType: "responses"
        )
        let obj = try JSONSerialization.jsonObject(with: Data(entry.utf8)) as? [String: Any]
        #expect(obj?["apiType"] as? String == "responses")
    }
}
