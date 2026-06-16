import Foundation
import HTTPTypes
import MLXZCore

extension APIError {
    var httpStatus: HTTPResponse.Status {
        switch kind {
        case .invalidRequest: return .badRequest
        case .authentication: return .unauthorized
        case .notFound:       return .notFound
        case .rateLimited:    return .tooManyRequests
        case .server:         return .internalServerError
        }
    }

    /// The OpenAI-format error body: `{ "error": { message, type, param, code } }`.
    func jsonBody() -> Data {
        var fields: [(String, OAIJSON)] = [
            ("message", .string(message)),
            ("type", .string(type)),
        ]
        fields.append(("param", param.map(OAIJSON.string) ?? .null))
        fields.append(("code", code.map(OAIJSON.string) ?? .null))
        let body: OAIJSON = .object([("error", .object(fields))])
        return (try? body.serialized()) ?? Data("{\"error\":{\"message\":\"internal error\"}}".utf8)
    }
}

/// Coerce any thrown error into an APIError (unknown errors become 500s).
func asAPIError(_ error: any Error) -> APIError {
    if let api = error as? APIError { return api }
    return APIError(kind: .server, message: String(describing: error), code: "internal_error")
}
