import Foundation
import Hummingbird
import HTTPTypes
import NIOCore
import MLXZCore

/// Rejects requests lacking a matching `Authorization: Bearer <apiKey>` header.
/// Optional — only added when the server is configured with an API key (e.g. for LAN binding).
struct APIKeyMiddleware<Context: RequestContext>: RouterMiddleware {
    let apiKey: String

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        // Allow the unauthenticated health check through.
        if request.uri.path == "/health" {
            return try await next(request, context)
        }
        let provided = request.headers[.authorization]
        let expected = "Bearer \(apiKey)"
        guard provided == expected else {
            let err = APIError(kind: .authentication, message: "Missing or invalid API key.", code: "invalid_api_key")
            return errorResponse(err)
        }
        return try await next(request, context)
    }
}
