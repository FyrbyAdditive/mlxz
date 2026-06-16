import Foundation

/// A domain error carrying everything needed to render an OpenAI-format error response.
/// The server's error middleware maps this to JSON + HTTP status; the core defines it so
/// engines and the server share one vocabulary.
public struct APIError: Error, Sendable {
    /// Maps to an HTTP status code in the server layer.
    public enum Kind: Sendable {
        case invalidRequest   // 400
        case authentication   // 401
        case notFound         // 404
        case rateLimited      // 429
        case server           // 500
    }

    public var kind: Kind
    public var message: String
    /// OpenAI `error.code`, e.g. "model_not_loaded".
    public var code: String?
    /// OpenAI `error.param`, when a specific field is at fault.
    public var param: String?

    public init(kind: Kind, message: String, code: String? = nil, param: String? = nil) {
        self.kind = kind
        self.message = message
        self.code = code
        self.param = param
    }

    /// OpenAI `error.type` string for this kind.
    public var type: String {
        switch kind {
        case .invalidRequest: return "invalid_request_error"
        case .authentication: return "authentication_error"
        case .notFound:       return "not_found_error"
        case .rateLimited:    return "rate_limit_error"
        case .server:         return "server_error"
        }
    }

    // Common cases.
    public static func noModelLoaded() -> APIError {
        APIError(kind: .notFound, message: "No model is currently loaded.", code: "model_not_loaded")
    }

    public static func unsupportedCapability(_ what: String) -> APIError {
        APIError(kind: .invalidRequest, message: "The loaded model does not support \(what).", code: "unsupported_capability")
    }

    public static func busy() -> APIError {
        APIError(kind: .rateLimited, message: "The server is busy generating another response.", code: "server_busy")
    }
}
