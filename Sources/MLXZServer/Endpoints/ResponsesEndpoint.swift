import Foundation
import MLXZCore

/// `/v1/responses` — the newer OpenAI surface. Copilot can target this via `apiType: responses`.
struct ResponsesEndpoint: OpenAIEndpoint {
    typealias WireRequest = ResponsesRequest

    static let path = "/v1/responses"
    static let requiredCapabilities: ModelCapabilities = [.chat]

    func isStreaming(_ wire: ResponsesRequest) -> Bool { wire.isStreaming }

    func toGenerationRequest(_ wire: ResponsesRequest, modelID: String) throws -> GenerationRequest {
        var messages: [ChatMessage] = []
        if let instructions = wire.instructions, !instructions.isEmpty {
            messages.append(ChatMessage(role: .system, text: instructions))
        }
        messages.append(contentsOf: try Self.mapInput(wire.input))

        var sampling = SamplingParameters()
        if let t = wire.temperature { sampling.temperature = t }
        if let p = wire.topP { sampling.topP = p }

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
            maxTokens: wire.maxOutputTokens,
            tools: (tools?.isEmpty == true) ? nil : tools
        )
    }

    static func mapInput(_ input: ResponsesInput) throws -> [ChatMessage] {
        switch input {
        case .text(let s):
            return [ChatMessage(role: .user, text: s)]
        case .items(let items):
            return items.map { item in
                let role = ChatMessage.Role(rawValue: item.role ?? "user") ?? .user
                var parts: [ContentPart] = []
                switch item.content {
                case .text(let s):
                    parts.append(.text(s))
                case .parts(let ps):
                    for p in ps {
                        if (p.type == "input_text" || p.type == "text"), let t = p.text {
                            parts.append(.text(t))
                        } else if (p.type == "input_image" || p.type == "image_url"),
                                  let urlStr = p.imageURL, let imagePart = ImageContent.part(fromURLString: urlStr) {
                            parts.append(imagePart)
                        }
                    }
                case .none:
                    break
                }
                return ChatMessage(role: role, content: parts)
            }
        }
    }

    func makeStreamEncoder(for wire: ResponsesRequest, modelID: String) -> any SSEEventEncoder {
        ResponsesStreamEncoder(modelID: modelID)
    }

    func encodeNonStreaming(_ result: AggregatedResult, wire: ResponsesRequest, modelID: String) throws -> Data {
        let responseID = "resp_\(OpenAIID.random())"
        let itemID = "msg_\(OpenAIID.random())"
        let hasText = !result.text.isEmpty

        let output = ResponsesPayload.outputItems(
            text: result.text, hasText: hasText, itemID: itemID, toolCalls: result.toolCalls)

        let body: OAIJSON = .object([
            ("id", .string(responseID)),
            ("object", .string("response")),
            ("created_at", .int(OpenAIID.timestamp())),
            ("status", .string("completed")),
            ("model", .string(modelID)),
            ("output", .array(output)),
            ("usage", .object([
                ("input_tokens", .int(result.usage.promptTokens)),
                ("output_tokens", .int(result.usage.completionTokens)),
                ("total_tokens", .int(result.usage.totalTokens)),
            ])),
        ])
        return try body.serialized()
    }
}
