import Foundation
import MLXZCore

/// Encodes the internal event stream as OpenAI `chat.completion.chunk` SSE frames,
/// terminated by a literal `data: [DONE]` frame.
final class ChatCompletionStreamEncoder: SSEEventEncoder, @unchecked Sendable {
    private let id: String
    private let modelID: String
    private let created: Int
    private let includeUsage: Bool
    private var sentRole = false
    private var toolCallIndex = 0

    init(modelID: String, includeUsage: Bool = false) {
        self.id = "chatcmpl-\(OpenAIID.random())"
        self.modelID = modelID
        self.created = OpenAIID.timestamp()
        self.includeUsage = includeUsage
    }

    func encode(_ event: GenerationEvent) throws -> [SSEFrame] {
        switch event {
        case .started:
            return []

        case .reasoningDelta(let text):
            // Surface chain-of-thought on the `reasoning_content` delta field (the de-facto
            // standard reasoning channel for OpenAI-compatible clients), kept out of `content`.
            var delta: [(String, OAIJSON)] = []
            if !sentRole {
                delta.append(("role", .string("assistant")))
                sentRole = true
            }
            delta.append(("reasoning_content", .string(text)))
            return [chunkFrame(delta: .object(delta), finishReason: nil)]

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
            // Final content chunk with finish_reason and an empty delta.
            var frames = [chunkFrame(delta: .object([]), finishReason: result.finishReason.rawValue)]
            // Per OpenAI: when stream_options.include_usage is set, follow with one extra chunk that
            // has an empty `choices` array and a populated `usage`. VS Code Copilot reads this to
            // drive its context-window % counter; without it the counter sits at 0%. All three token
            // fields are required by Copilot's usage type-guard.
            if includeUsage {
                frames.append(usageFrame(result.usage))
            }
            return frames
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

    /// The OpenAI-standard trailing usage chunk: empty `choices`, populated `usage`.
    private func usageFrame(_ usage: TokenUsage) -> SSEFrame {
        let chunk: OAIJSON = .object([
            ("id", .string(id)),
            ("object", .string("chat.completion.chunk")),
            ("created", .int(created)),
            ("model", .string(modelID)),
            ("choices", .array([])),
            ("usage", .object([
                ("prompt_tokens", .int(usage.promptTokens)),
                ("completion_tokens", .int(usage.completionTokens)),
                ("total_tokens", .int(usage.totalTokens)),
            ])),
        ])
        return SSEFrame(event: nil, data: chunk.jsonString)
    }
}
