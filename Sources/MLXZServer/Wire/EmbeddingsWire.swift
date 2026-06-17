import Foundation
import MLXZCore

/// The OpenAI `/v1/embeddings` request. `input` may be a string or an array of strings.
struct EmbeddingsRequest: Decodable, Sendable {
    var model: String?
    var input: EmbeddingsInput
}

enum EmbeddingsInput: Decodable, Sendable {
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

/// Builds the OpenAI embeddings response body.
enum EmbeddingsResponse {
    static func json(_ result: EmbeddingResult, model: String) throws -> Data {
        let data: [OAIJSON] = result.vectors.enumerated().map { idx, vector in
            .object([
                ("object", .string("embedding")),
                ("index", .int(idx)),
                ("embedding", .array(vector.map { OAIJSON.double(Double($0)) })),
            ])
        }
        let body: OAIJSON = .object([
            ("object", .string("list")),
            ("data", .array(data)),
            ("model", .string(model)),
            ("usage", .object([
                ("prompt_tokens", .int(result.promptTokens)),
                ("total_tokens", .int(result.promptTokens)),
            ])),
        ])
        return try body.serialized()
    }
}
