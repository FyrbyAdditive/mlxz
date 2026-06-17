import Foundation
import MLXZCore

/// Encodes the internal event stream as legacy `text_completion` SSE chunks
/// (`choices[].text` deltas), terminated by `data: [DONE]`.
final class TextCompletionStreamEncoder: SSEEventEncoder, @unchecked Sendable {
    private let id: String
    private let modelID: String
    private let created: Int

    init(modelID: String) {
        self.id = "cmpl-\(OpenAIID.random())"
        self.modelID = modelID
        self.created = OpenAIID.timestamp()
    }

    func encode(_ event: GenerationEvent) throws -> [SSEFrame] {
        switch event {
        case .started, .toolCall:
            return []  // legacy completions has no tool calls
        case .textDelta(let text):
            return [chunk(text: text, finishReason: nil)]
        case .completed(let result):
            return [chunk(text: "", finishReason: result.finishReason.rawValue)]
        }
    }

    func terminator() -> [SSEFrame] { [SSEFrame(event: nil, data: "[DONE]")] }

    private func chunk(text: String, finishReason: String?) -> SSEFrame {
        let body: OAIJSON = .object([
            ("id", .string(id)),
            ("object", .string("text_completion")),
            ("created", .int(created)),
            ("model", .string(modelID)),
            ("choices", .array([
                .object([
                    ("text", .string(text)),
                    ("index", .int(0)),
                    ("finish_reason", finishReason.map(OAIJSON.string) ?? .null),
                ]),
            ])),
        ])
        return SSEFrame(event: nil, data: body.jsonString)
    }
}
