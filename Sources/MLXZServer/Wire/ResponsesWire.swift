import Foundation
import MLXZCore

/// The subset of the OpenAI Responses request we accept. `input` may be a plain string or an
/// array of input items (messages with typed content), mirroring the Responses API.
struct ResponsesRequest: Decodable, Sendable {
    var model: String?
    var input: ResponsesInput
    var instructions: String?
    var temperature: Float?
    var topP: Float?
    var maxOutputTokens: Int?
    var stream: Bool?
    var tools: [WireTool]?

    enum CodingKeys: String, CodingKey {
        case model, input, instructions, temperature, stream, tools
        case topP = "top_p"
        case maxOutputTokens = "max_output_tokens"
    }

    var isStreaming: Bool { stream ?? false }
}

/// `input` is either a bare string or an array of input items.
enum ResponsesInput: Decodable, Sendable {
    case text(String)
    case items([ResponsesInputItem])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .text(s)
        } else {
            self = .items(try container.decode([ResponsesInputItem].self))
        }
    }
}

/// One input item — a message with a role and content (string or typed parts).
struct ResponsesInputItem: Decodable, Sendable {
    var role: String?
    var content: ResponsesItemContent?
    var type: String?
}

/// Item content: a bare string or an array of typed input parts (input_text / input_image).
enum ResponsesItemContent: Decodable, Sendable {
    case text(String)
    case parts([ResponsesContentPart])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .text(s)
        } else {
            self = .parts(try container.decode([ResponsesContentPart].self))
        }
    }
}

struct ResponsesContentPart: Decodable, Sendable {
    var type: String
    var text: String?
    var imageURL: String?

    enum CodingKeys: String, CodingKey {
        case type, text
        case imageURL = "image_url"
    }
}
