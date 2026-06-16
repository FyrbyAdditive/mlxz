import Foundation
import Hummingbird
import HTTPTypes
import NIOCore
import Logging
import MLXZCore

/// Builds the OpenAI-compatible router. Extracted from `InferenceServer` so tests can drive it
/// directly via `app.test(.router)` with a mock-loaded `ModelManager` and no real network.
struct RouterBuilder {
    let manager: ModelManager
    let gate: GenerationGate
    let apiKey: String?
    let logSink: (@Sendable (String) -> Void)?

    func build() -> Router<BasicRequestContext> {
        let router = Router()
        router.add(middleware: LogRequestsMiddleware(.info))
        if let apiKey {
            router.add(middleware: APIKeyMiddleware(apiKey: apiKey))
        }

        register(ChatCompletionsEndpoint(), on: router)
        // Future endpoints land here with one line each:
        // register(ResponsesEndpoint(), on: router)
        // register(ModelsListEndpoint(), on: router)

        router.get("/health") { _, _ in
            Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(string: #"{"status":"ok"}"#))
            )
        }
        return router
    }

    /// The generic handler shared by every endpoint: decode → validate → translate →
    /// branch stream/non-stream. Adding an endpoint never touches this code.
    private func register<E: OpenAIEndpoint>(_ endpoint: E, on router: Router<BasicRequestContext>) {
        let manager = self.manager
        let gate = self.gate
        let logSink = self.logSink

        router.post(RouterPath(stringLiteral: E.path)) { request, context -> Response in
            let wire: E.WireRequest
            do {
                let bytes = try await request.body.collect(upTo: 64 * 1024 * 1024)
                let data = Data(buffer: bytes)
                wire = try JSONDecoder().decode(E.WireRequest.self, from: data)
            } catch let api as APIError {
                return errorResponse(api)
            } catch {
                return errorResponse(APIError(kind: .invalidRequest, message: "Malformed request body: \(error)", code: "invalid_body"))
            }

            guard let engine = await manager.currentEngine() else {
                return errorResponse(.noModelLoaded())
            }
            guard engine.capabilities.contains(E.requiredCapabilities) else {
                return errorResponse(.unsupportedCapability("this endpoint"))
            }

            let modelID = engine.descriptor.repoID
            let genRequest: GenerationRequest
            do {
                genRequest = try endpoint.toGenerationRequest(wire, modelID: modelID)
            } catch let api as APIError {
                return errorResponse(api)
            } catch {
                return errorResponse(asAPIError(error))
            }

            guard await gate.tryAcquire() else {
                return errorResponse(.busy())
            }

            logSink?("→ \(E.path) (\(genRequest.messages.count) messages, stream=\(endpoint.isStreaming(wire)))")

            if endpoint.isStreaming(wire) {
                return makeStreamingResponse(endpoint: endpoint, wire: wire, modelID: modelID, engine: engine, request: genRequest, gate: gate)
            } else {
                do {
                    let stream = try await engine.generate(genRequest)
                    let aggregated = try await AggregatedResult.collect(from: stream)
                    let data = try endpoint.encodeNonStreaming(aggregated, wire: wire, modelID: modelID)
                    await gate.release()
                    return Response(
                        status: .ok,
                        headers: [.contentType: "application/json"],
                        body: .init(byteBuffer: ByteBuffer(data: data))
                    )
                } catch {
                    await gate.release()
                    return errorResponse(asAPIError(error))
                }
            }
        }
    }
}

/// Build a streaming SSE response. Free function: touches no actor state, so it is safely
/// callable from inside the (Sendable) route closure.
private func makeStreamingResponse<E: OpenAIEndpoint>(
    endpoint: E,
    wire: E.WireRequest,
    modelID: String,
    engine: any InferenceEngine,
    request: GenerationRequest,
    gate: GenerationGate
) -> Response {
    let encoder = endpoint.makeStreamEncoder(for: wire, modelID: modelID)
    let headers: HTTPFields = [
        .contentType: "text/event-stream",
        .cacheControl: "no-cache",
        .connection: "keep-alive",
    ]
    return Response(
        status: .ok,
        headers: headers,
        body: ResponseBody { writer in
            let allocator = ByteBufferAllocator()
            do {
                let stream = try await engine.generate(request)
                for try await event in stream {
                    for frame in try encoder.encode(event) {
                        try await writer.write(frame.byteBuffer(allocator: allocator))
                    }
                }
                for frame in encoder.terminator() {
                    try await writer.write(frame.byteBuffer(allocator: allocator))
                }
                await gate.release()
                try await writer.finish(nil)
            } catch {
                // Mid-stream errors: best-effort emit an SSE error frame, then finish.
                let api = asAPIError(error)
                let errData = String(data: api.jsonBody(), encoding: .utf8) ?? "{}"
                let frame = SSEFrame(event: nil, data: errData)
                try? await writer.write(frame.byteBuffer(allocator: allocator))
                await gate.release()
                try await writer.finish(nil)
            }
        }
    )
}

/// Build an OpenAI-format error response.
func errorResponse(_ error: APIError) -> Response {
    Response(
        status: error.httpStatus,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(data: error.jsonBody()))
    )
}
