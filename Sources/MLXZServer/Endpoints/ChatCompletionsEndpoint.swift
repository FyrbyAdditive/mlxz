import Foundation
import MLXZCore

/// `/v1/chat/completions` — the endpoint VS Code Copilot BYOK uses by default.
struct ChatCompletionsEndpoint: OpenAIEndpoint {
    typealias WireRequest = ChatCompletionRequest

    static let path = "/v1/chat/completions"
    static let requiredCapabilities: ModelCapabilities = [.chat]

    func isStreaming(_ wire: ChatCompletionRequest) -> Bool { wire.isStreaming }

    func toGenerationRequest(_ wire: ChatCompletionRequest, modelID: String) throws -> GenerationRequest {
        let messages = try wire.messages.map(Self.mapMessage)
        var sampling = SamplingParameters()
        if let t = wire.temperature { sampling.temperature = t }
        if let p = wire.topP { sampling.topP = p }
        if let s = wire.seed { sampling.seed = s }

        let tools = wire.tools?.map { tool in
            ToolDefinition(
                name: tool.function.name,
                description: tool.function.description,
                parametersJSONSchema: tool.function.parameters?.text
            )
        }

        return GenerationRequest(
            messages: messages,
            sampling: sampling,
            maxTokens: wire.maxCompletionTokens ?? wire.maxTokens,
            stop: wire.stop?.values ?? [],
            tools: (tools?.isEmpty == true) ? nil : tools,
            reasoningTokenBudget: wire.resolvedReasoningBudget
        )
    }

    static func mapMessage(_ m: WireMessage) throws -> ChatMessage {
        guard let role = ChatMessage.Role(rawValue: m.role) else {
            throw APIError(kind: .invalidRequest, message: "Unknown message role '\(m.role)'.", code: "invalid_role", param: "messages.role")
        }
        var parts: [ContentPart] = []
        switch m.content {
        case .text(let s):
            parts.append(.text(s))
        case .parts(let ps):
            for p in ps {
                if p.type == "text", let t = p.text {
                    parts.append(.text(t))
                } else if p.type == "image_url", let urlStr = p.imageURL?.url,
                          let imagePart = ImageContent.part(fromURLString: urlStr) {
                    parts.append(imagePart)
                }
            }
        case .null, .none:
            break
        }
        let toolCalls: [ToolCall] = (m.toolCalls ?? []).compactMap { tc in
            guard let fn = tc.function, let name = fn.name else { return nil }
            return ToolCall(id: tc.id ?? "call", name: name, argumentsJSON: fn.arguments ?? "{}")
        }
        return ChatMessage(role: role, content: parts, toolCalls: toolCalls, toolCallID: m.toolCallID)
    }

    func makeStreamEncoder(for wire: ChatCompletionRequest, modelID: String) -> any SSEEventEncoder {
        ChatCompletionStreamEncoder(modelID: modelID, includeUsage: wire.includeUsage)
    }

    func encodeNonStreaming(_ result: AggregatedResult, wire: ChatCompletionRequest, modelID: String) throws -> Data {
        let id = "chatcmpl-\(OpenAIID.random())"
        let created = OpenAIID.timestamp()

        var message: [(String, OAIJSON)] = [("role", .string("assistant"))]
        let content: OAIJSON = (result.text.isEmpty && !result.toolCalls.isEmpty) ? .null : .string(result.text)
        message.append(("content", content))
        if !result.reasoning.isEmpty {
            message.append(("reasoning_content", .string(result.reasoning)))
        }
        if !result.toolCalls.isEmpty {
            let calls: [OAIJSON] = result.toolCalls.enumerated().map { idx, call in
                .object([
                    ("index", .int(idx)),
                    ("id", .string(call.id)),
                    ("type", .string("function")),
                    ("function", .object([
                        ("name", .string(call.name)),
                        ("arguments", .string(call.argumentsJSON)),
                    ])),
                ])
            }
            message.append(("tool_calls", .array(calls)))
        }

        let body: OAIJSON = .object([
            "id": .string(id),
            "object": .string("chat.completion"),
            "created": .int(created),
            "model": .string(modelID),
            "choices": .array([
                .object([
                    "index": .int(0),
                    "message": .object(message),
                    "finish_reason": .string(result.finishReason.rawValue),
                ]),
            ]),
            "usage": .object([
                "prompt_tokens": .int(result.usage.promptTokens),
                "completion_tokens": .int(result.usage.completionTokens),
                "total_tokens": .int(result.usage.totalTokens),
            ]),
        ])
        return try body.serialized()
    }
}
