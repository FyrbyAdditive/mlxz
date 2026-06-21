import Foundation
import Testing
import MLXZCore

@testable import MLXZInference

/// The tool-history (`messages:` dict) path must carry images, not drop them — otherwise an image sent
/// in an agentic turn (any conversation that already has a tool call/result) is invisible to the model
/// ("I cannot see an image"). `mapMessageDict` should emit `{"type":"image"}` content parts, and
/// `images(from:)` should extract the pixels to pass via UserInput(messages:images:).
@Suite struct MessageMappingTests {
    private func redPNG() -> Data {
        // 1x1 red PNG (enough for CIImage to decode).
        Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")!
    }

    @Test func mapMessageDictEmitsImageParts() {
        let msg = ChatMessage(role: .user, content: [
            .text("What is this?"),
            .imageData(redPNG()),
        ])
        let dict = MLXInferenceEngine.mapMessageDict(msg)
        let content = dict["content"] as? [[String: any Sendable]]
        #expect(content != nil, "content should be an array of parts when an image is present")
        let types = content?.compactMap { $0["type"] as? String } ?? []
        #expect(types.contains("image"))
        #expect(types.contains("text"))
    }

    @Test func mapMessageDictPlainTextStaysString() {
        let msg = ChatMessage(role: .user, content: [.text("hello")])
        let dict = MLXInferenceEngine.mapMessageDict(msg)
        #expect(dict["content"] as? String == "hello")
    }

    @Test func imagesAreExtractedFromMessage() {
        let msg = ChatMessage(role: .user, content: [.text("hi"), .imageData(redPNG())])
        let imgs = MLXInferenceEngine.images(from: msg, maxImagePixels: 0)
        #expect(imgs.count == 1)
    }

    @Test func toolCallsAndImagesCoexistInDict() {
        // An assistant turn shouldn't normally carry images, but tool fields must survive regardless.
        let msg = ChatMessage(
            role: .assistant, content: [.text("")],
            toolCalls: [ToolCall(id: "c1", name: "f", argumentsJSON: "{}")])
        let dict = MLXInferenceEngine.mapMessageDict(msg)
        #expect(dict["tool_calls"] != nil)
    }
}
