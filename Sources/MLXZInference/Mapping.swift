import Foundation
import CoreImage
import MLXZCore
import MLXLMCommon

extension MLXInferenceEngine {
    /// Map our normalized `ChatMessage` to the package's `Chat.Message`.
    static func mapMessage(_ message: ChatMessage) -> Chat.Message {
        let role: Chat.Message.Role = switch message.role {
        case .system: .system
        case .user: .user
        case .assistant: .assistant
        case .tool: .tool
        }

        let text = message.content.compactMap { part -> String? in
            if case let .text(t) = part { return t }
            return nil
        }.joined()

        let images: [UserInput.Image] = message.content.compactMap { part in
            switch part {
            case .imageURL(let url):
                return .url(url)
            case .imageData(let data):
                // Decode inline (base64 data-URL) image bytes into a CIImage for VLMs.
                return CIImage(data: data).map { UserInput.Image.ciImage($0) }
            case .text:
                return nil
            }
        }

        return Chat.Message(role: role, content: text, images: images, videos: [])
    }

    /// Map our `ChatMessage` to an OpenAI-shaped message dictionary for the chat template's
    /// `.messages` path. Unlike `Chat.Message` (which has no tool fields and silently drops them),
    /// this preserves assistant `tool_calls` and `tool` results — so a multi-turn agentic
    /// conversation is replayed faithfully (the template renders `<tool_call>`/`<tool_response>`).
    /// Without this the model sees a broken history and bails after ~2 tokens.
    static func mapMessageDict(_ message: ChatMessage) -> [String: any Sendable] {
        let roleString: String = switch message.role {
        case .system: "system"
        case .user: "user"
        case .assistant: "assistant"
        case .tool: "tool"
        }
        var dict: [String: any Sendable] = ["role": roleString]
        // Text content (image parts aren't supported on the tool/agentic path).
        dict["content"] = message.content.compactMap { part -> String? in
            if case let .text(t) = part { return t }
            return nil
        }.joined()

        if !message.toolCalls.isEmpty {
            dict["tool_calls"] = message.toolCalls.map { call -> [String: any Sendable] in
                // arguments must be a JSON object for the template's `arguments|items` iteration.
                let argsObj: any Sendable =
                    (call.argumentsJSON.data(using: .utf8)
                        .flatMap { try? JSONSerialization.jsonObject(with: $0) }
                        as? [String: any Sendable]) ?? [:]
                return [
                    "id": call.id, "type": "function",
                    "function": ["name": call.name, "arguments": argsObj],
                ]
            }
        }
        if let toolCallID = message.toolCallID {
            dict["tool_call_id"] = toolCallID
        }
        return dict
    }

    /// Map our `ToolDefinition` to the package's OpenAI-shaped `ToolSpec` dictionary.
    static func mapTool(_ tool: ToolDefinition) -> ToolSpec {
        var function: [String: any Sendable] = ["name": tool.name]
        if let description = tool.description {
            function["description"] = description
        }
        if let schema = tool.parametersJSONSchema,
           let data = schema.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: any Sendable] {
            function["parameters"] = obj
        }
        return ["type": "function", "function": function]
    }

    /// Map our `SamplingParameters` onto the package's `GenerateParameters`, applying perf options.
    static func mapParameters(
        _ sampling: SamplingParameters,
        maxTokens: Int?,
        perf: EnginePerfOptions = .default
    ) -> GenerateParameters {
        var params = GenerateParameters(
            maxTokens: maxTokens,
            maxKVSize: perf.maxKVSize,
            kvBits: perf.kvBits,
            kvGroupSize: perf.kvGroupSize,
            quantizedKVStart: perf.quantizedKVStart,
            temperature: sampling.temperature,
            topP: sampling.topP
        )
        if let topK = sampling.topK { params.topK = topK }
        if let rp = sampling.repetitionPenalty { params.repetitionPenalty = rp }
        return params
    }

    /// Map the package's native `ToolCall` to ours, serializing arguments back to JSON text.
    static func mapNativeToolCall(_ call: MLXLMCommon.ToolCall) -> MLXZCore.ToolCall {
        let argumentsJSON: String
        if let data = try? JSONEncoder().encode(call.function.arguments),
           let str = String(data: data, encoding: .utf8) {
            argumentsJSON = str
        } else {
            argumentsJSON = "{}"
        }
        return MLXZCore.ToolCall(
            id: "call_\(abs(call.hashValue))",
            name: call.function.name,
            argumentsJSON: argumentsJSON
        )
    }

    /// Map the package stop reason to ours. Tool calls override to `.toolCalls`.
    static func mapStopReason(_ reason: GenerateStopReason, sawToolCall: Bool) -> FinishReason {
        if sawToolCall { return .toolCalls }
        switch reason {
        case .stop: return .stop
        case .length: return .length
        case .cancelled: return .cancelled
        @unknown default: return .stop
        }
    }
}
