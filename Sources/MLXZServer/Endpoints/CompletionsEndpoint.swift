import Foundation
import MLXZCore

/// Legacy `/v1/completions` — a bare prompt → text completion. No chat template is applied;
/// the prompt is sent as a single user message (the closest faithful mapping for instruct models).
struct CompletionsEndpoint: OpenAIEndpoint {
    typealias WireRequest = CompletionRequest

    static let path = "/v1/completions"
    static let requiredCapabilities: ModelCapabilities = [.chat]

    func requestedModel(_ wire: CompletionRequest) -> String? { wire.model }
    func isStreaming(_ wire: CompletionRequest) -> Bool { wire.stream ?? false }

    func toGenerationRequest(_ wire: CompletionRequest, modelID: String) throws -> GenerationRequest {
        var sampling = SamplingParameters()
        if let t = wire.temperature { sampling.temperature = t }
        if let p = wire.topP { sampling.topP = p }
        return GenerationRequest(
            messages: [ChatMessage(role: .user, text: wire.prompt.text)],
            sampling: sampling,
            maxTokens: wire.maxTokens,
            stop: wire.stop?.values ?? []
        )
    }

    func makeStreamEncoder(for wire: CompletionRequest, modelID: String) -> any SSEEventEncoder {
        TextCompletionStreamEncoder(modelID: modelID)
    }

    func encodeNonStreaming(_ result: AggregatedResult, wire: CompletionRequest, modelID: String) throws -> Data {
        let body: OAIJSON = .object([
            ("id", .string("cmpl-\(OpenAIID.random())")),
            ("object", .string("text_completion")),
            ("created", .int(OpenAIID.timestamp())),
            ("model", .string(modelID)),
            ("choices", .array([
                .object([
                    ("text", .string(result.text)),
                    ("index", .int(0)),
                    ("finish_reason", .string(result.finishReason.rawValue)),
                ]),
            ])),
            ("usage", .object([
                ("prompt_tokens", .int(result.usage.promptTokens)),
                ("completion_tokens", .int(result.usage.completionTokens)),
                ("total_tokens", .int(result.usage.totalTokens)),
            ])),
        ])
        return try body.serialized()
    }
}

/// Legacy completions wire request. `prompt` may be a string or an array of strings (first used).
struct CompletionRequest: Decodable, Sendable {
    var model: String?
    var prompt: CompletionPrompt
    var temperature: Float?
    var topP: Float?
    var maxTokens: Int?
    var stop: StopValue?
    var stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model, prompt, temperature, stop, stream
        case topP = "top_p"
        case maxTokens = "max_tokens"
    }
}

enum CompletionPrompt: Decodable, Sendable {
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

    var text: String {
        switch self {
        case .one(let s): return s
        case .many(let a): return a.first ?? ""
        }
    }
}
