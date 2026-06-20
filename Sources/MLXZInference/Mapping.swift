import Foundation
import CoreImage
import MLXZCore
import MLXLMCommon

extension MLXInferenceEngine {
    /// Map our normalized `ChatMessage` to the package's `Chat.Message`.
    /// - Parameter maxImagePixels: downscale images above this pixel budget (aspect preserved) before
    ///   the vision encoder. A 24.5 MP photo otherwise makes the VLM allocate tens of GB for the
    ///   vision-token attention tensors — over the GPU's max Metal buffer — and hard-crashes the
    ///   process. 0 = no cap.
    static func mapMessage(_ message: ChatMessage, maxImagePixels: Int = 0) -> Chat.Message {
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
                // Local file URLs can be loaded + capped now; remote URLs are fetched by the
                // processor later (left uncapped — the data-URL path is the common upload route).
                if url.isFileURL, let ci = CIImage(contentsOf: url) {
                    return .ciImage(downscale(ci, maxPixels: maxImagePixels))
                }
                return .url(url)
            case .imageData(let data):
                // Decode inline (base64 data-URL) image bytes into a CIImage for VLMs, capping size.
                return CIImage(data: data).map { .ciImage(downscale($0, maxPixels: maxImagePixels)) }
            case .text:
                return nil
            }
        }

        return Chat.Message(role: role, content: text, images: images, videos: [])
    }

    /// Downscale a CIImage so width×height ≤ `maxPixels` (aspect preserved). No-op when `maxPixels`
    /// is 0 or the image is already within budget. Guards the VLM against multi-GB vision-token
    /// tensors that exceed the GPU's max Metal buffer.
    static func downscale(_ image: CIImage, maxPixels: Int) -> CIImage {
        guard maxPixels > 0 else { return image }
        let w = image.extent.width, h = image.extent.height
        guard w > 0, h > 0 else { return image }
        let pixels = w * h
        guard pixels > CGFloat(maxPixels) else { return image }
        let scale = (CGFloat(maxPixels) / pixels).squareRoot()
        return image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
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
    ///
    /// `repoID` is used to suppress KV-cache quantization for models whose attention uses sinks
    /// (e.g. gpt-oss): the quantized-attention kernel has no sink support and hard-`fatalError`s on
    /// the first decode step, crashing the process. For those models we force a full-precision cache.
    static func mapParameters(
        _ sampling: SamplingParameters,
        maxTokens: Int?,
        perf: EnginePerfOptions = .default,
        repoID: String = ""
    ) -> GenerateParameters {
        // Disable KV quantization for sink-attention models — the fork's quantized SDPA path is
        // unimplemented for non-zero sinks and traps. Full precision is correct (and the attention KV
        // is a minor memory fraction anyway), so this is a safe, model-targeted fallback.
        let kvBits = usesAttentionSinks(repoID) ? nil : perf.kvBits
        var params = GenerateParameters(
            maxTokens: maxTokens,
            maxKVSize: perf.maxKVSize,
            kvBits: kvBits,
            kvGroupSize: perf.kvGroupSize,
            quantizedKVStart: perf.quantizedKVStart,
            temperature: sampling.temperature,
            topP: sampling.topP
        )
        if let topK = sampling.topK { params.topK = topK }
        if let rp = sampling.repetitionPenalty { params.repetitionPenalty = rp }
        return params
    }

    /// Whether a model's attention uses sinks (a learned per-head bias added to the softmax). The
    /// quantized KV-cache attention kernel does not support sinks and traps if asked to, so KV
    /// quantization must be disabled for these. Currently the gpt-oss family.
    static func usesAttentionSinks(_ repoID: String) -> Bool {
        let id = repoID.lowercased()
        return id.contains("gpt-oss") || id.contains("gpt_oss") || id.contains("gptoss")
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
