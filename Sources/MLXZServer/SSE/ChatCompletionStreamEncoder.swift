import Foundation
import MLXZCore

/// Encodes the internal event stream as OpenAI `chat.completion.chunk` SSE frames,
/// terminated by a literal `data: [DONE]` frame.
final class ChatCompletionStreamEncoder: SSEEventEncoder, @unchecked Sendable {
    private let id: String
    private let modelID: String
    private let created: Int
    private var sentRole = false
    private var toolCallIndex = 0

    init(modelID: String) {
        self.id = "chatcmpl-\(OpenAIID.random())"
        self.modelID = modelID
        self.created = OpenAIID.timestamp()
    }

    func encode(_ event: GenerationEvent) throws -> [SSEFrame] {
        switch event {
        case .started:
            return []

        case .textDelta(let text):
            var delta: [(String, OAIJSON)] = []
            if !sentRole {
                delta.append(("role", .string("assistant")))
                sentRole = true
            }
            delta.append(("content", .string(text)))
            return [chunkFrame(delta: .object(delta), finishReason: nil)]

        case .toolCall(let call):
            // Emit a single chunk carrying the full tool call (arguments in one piece).
            let delta: OAIJSON = .object([
                ("tool_calls", .array([
                    .object([
                        ("index", .int(toolCallIndex)),
                        ("id", .string(call.id)),
                        ("type", .string("function")),
                        ("function", .object([
                            ("name", .string(call.name)),
                            ("arguments", .string(call.argumentsJSON)),
                        ])),
                    ]),
                ])),
            ])
            toolCallIndex += 1
            return [chunkFrame(delta: delta, finishReason: nil)]

        case .completed(let result):
            // Final chunk with finish_reason and an empty delta.
            return [chunkFrame(delta: .object([]), finishReason: result.finishReason.rawValue)]
        }
    }

    func terminator() -> [SSEFrame] {
        [SSEFrame(event: nil, data: "[DONE]")]
    }

    private func chunkFrame(delta: OAIJSON, finishReason: String?) -> SSEFrame {
        let choice: OAIJSON = .object([
            ("index", .int(0)),
            ("delta", delta),
            ("finish_reason", finishReason.map(OAIJSON.string) ?? .null),
        ])
        let chunk: OAIJSON = .object([
            ("id", .string(id)),
            ("object", .string("chat.completion.chunk")),
            ("created", .int(created)),
            ("model", .string(modelID)),
            ("choices", .array([choice])),
        ])
        return SSEFrame(event: nil, data: chunk.jsonString)
    }
}
