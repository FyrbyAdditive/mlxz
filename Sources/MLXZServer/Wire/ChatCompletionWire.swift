import Foundation
import MLXZCore

// MARK: - Request

/// The subset of the OpenAI Chat Completions request we accept. Lenient by design:
/// unknown fields are ignored, `content` may be a string or an array of parts.
struct ChatCompletionRequest: Decodable, Sendable {
    var model: String?
    var messages: [WireMessage]
    var temperature: Float?
    var topP: Float?
    var maxTokens: Int?
    var maxCompletionTokens: Int?
    var stop: StopValue?
    var stream: Bool?
    var streamOptions: StreamOptions?
    var tools: [WireTool]?
    var seed: UInt64?
    /// OpenAI `reasoning_effort` ("low" | "medium" | "high"). Mapped to a reasoning-token budget.
    var reasoningEffort: String?
    /// Explicit numeric override of the reasoning-token budget. We accept both the OpenAI-ish
    /// `max_reasoning_tokens` and Qwen's `thinking_budget` spelling; either wins over `reasoning_effort`.
    var maxReasoningTokens: Int?
    var thinkingBudget: Int?

    enum CodingKeys: String, CodingKey {
        case model, messages, temperature, stop, stream, tools, seed
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case streamOptions = "stream_options"
        case reasoningEffort = "reasoning_effort"
        case maxReasoningTokens = "max_reasoning_tokens"
        case thinkingBudget = "thinking_budget"
    }

    /// Resolve the per-request reasoning-token budget from the wire fields, or nil to fall back to
    /// the engine default. An explicit numeric field (`max_reasoning_tokens`/`thinking_budget`) wins;
    /// otherwise map `reasoning_effort`. Values: low→256, medium→1024, high→4096, "none"/"minimal"→1.
    /// A negative number means "uncapped" (≤0 disables the cap in the engine).
    var resolvedReasoningBudget: Int? {
        if let n = maxReasoningTokens ?? thinkingBudget { return n }
        switch reasoningEffort?.lowercased() {
        case "none", "minimal": return 1
        case "low": return 256
        case "medium": return 1024
        case "high": return 4096
        default: return nil
        }
    }

    var isStreaming: Bool { stream ?? false }
    /// OpenAI `stream_options.include_usage`: when true, emit a final `choices:[]` chunk carrying
    /// `usage`. VS Code Copilot sets this and uses the reported `prompt_tokens` to drive its
    /// context-window % counter.
    var includeUsage: Bool { streamOptions?.includeUsage ?? false }
}

/// OpenAI `stream_options` object.
struct StreamOptions: Decodable, Sendable {
    var includeUsage: Bool?
    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

/// `stop` may be a single string or an array of strings.
enum StopValue: Decodable, Sendable {
    case one(String)
    case many([String])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .one(s)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    var values: [String] {
        switch self {
        case .one(let s): return [s]
        case .many(let a): return a
        }
    }
}

struct WireMessage: Decodable, Sendable {
    var role: String
    var content: WireContent?
    var toolCallID: String?
    var toolCalls: [WireToolCall]?

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCallID = "tool_call_id"
        case toolCalls = "tool_calls"
    }
}

/// `content` may be a plain string or an array of typed parts (text / image_url).
enum WireContent: Decodable, Sendable {
    case text(String)
    case parts([WirePart])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let s = try? container.decode(String.self) {
            self = .text(s)
        } else {
            self = .parts(try container.decode([WirePart].self))
        }
    }
}

struct WirePart: Decodable, Sendable {
    var type: String
    var text: String?
    var imageURL: WireImageURL?

    enum CodingKeys: String, CodingKey {
        case type, text
        case imageURL = "image_url"
    }
}

struct WireImageURL: Decodable, Sendable {
    var url: String
}

struct WireTool: Decodable, Sendable {
    var type: String?
    var function: WireFunction
}

struct WireFunction: Decodable, Sendable {
    var name: String
    var description: String?
    /// Arbitrary JSON Schema; captured as raw text to forward unchanged.
    var parameters: RawJSON?
}

struct WireToolCall: Decodable, Sendable {
    var id: String?
    var function: WireToolCallFunction?
}

struct WireToolCallFunction: Decodable, Sendable {
    var name: String?
    var arguments: String?
}

/// Captures an arbitrary JSON value as its serialized text, for pass-through fields.
struct RawJSON: Decodable, Sendable {
    var text: String
    init(from decoder: Decoder) throws {
        let value = try AnyCodable(from: decoder)
        let data = try JSONSerialization.data(withJSONObject: value.value, options: [])
        self.text = String(data: data, encoding: .utf8) ?? "{}"
    }
}

/// Minimal `Any`-backed Codable used only to round-trip opaque JSON Schema.
struct AnyCodable: Decodable {
    let value: Any
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            value = NSNull()
        } else if let b = try? container.decode(Bool.self) {
            value = b
        } else if let i = try? container.decode(Int.self) {
            value = i
        } else if let d = try? container.decode(Double.self) {
            value = d
        } else if let s = try? container.decode(String.self) {
            value = s
        } else if let arr = try? container.decode([AnyCodable].self) {
            value = arr.map(\.value)
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            value = dict.mapValues(\.value)
        } else {
            value = NSNull()
        }
    }
}
