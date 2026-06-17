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
    /// Optional source of additional servable model ids for `/v1/models` (e.g. installed models).
    /// The currently-loaded model is always included.
    var extraModelIDs: (@Sendable () -> [String])?
    /// Optional embedding manager; when set, `/v1/embeddings` is served.
    var embeddingManager: EmbeddingManager?
    /// Optional sink fired once per completed generation with its token usage (for UI metrics).
    var metricsSink: (@Sendable (TokenUsage) -> Void)?

    func build() -> Router<BasicRequestContext> {
        let router = Router()
        router.add(middleware: LogRequestsMiddleware(.info))
        if let apiKey {
            router.add(middleware: APIKeyMiddleware(apiKey: apiKey))
        }

        register(ChatCompletionsEndpoint(), on: router)
        register(ResponsesEndpoint(), on: router)
        register(CompletionsEndpoint(), on: router)

        if let embeddingManager {
            registerEmbeddings(embeddingManager, on: router)
        }

        router.get("/health") { _, _ in
            Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(string: #"{"status":"ok"}"#))
            )
        }

        // GET /v1/models — lists the loaded model plus any extra servable ids.
        let manager = self.manager
        let extraModelIDs = self.extraModelIDs
        router.get("/v1/models") { _, _ -> Response in
            var ids: [String] = []
            if let loaded = await manager.currentEngine()?.descriptor.repoID {
                ids.append(loaded)
            }
            ids.append(contentsOf: extraModelIDs?() ?? [])
            // De-dup preserving order.
            var seen = Set<String>()
            let unique = ids.filter { seen.insert($0).inserted }
            let data = (try? Self.modelsListJSON(unique)) ?? Data("{\"object\":\"list\",\"data\":[]}".utf8)
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(data: data))
            )
        }
        return router
    }

    /// POST /v1/embeddings — embeds input text(s) using the requested model.
    private func registerEmbeddings(_ manager: EmbeddingManager, on router: Router<BasicRequestContext>) {
        router.post("/v1/embeddings") { request, _ -> Response in
            let wire: EmbeddingsRequest
            do {
                let bytes = try await request.body.collect(upTo: 16 * 1024 * 1024)
                wire = try JSONDecoder().decode(EmbeddingsRequest.self, from: Data(buffer: bytes))
            } catch {
                return errorResponse(APIError(kind: .invalidRequest, message: "Malformed request body: \(error)", code: "invalid_body"))
            }
            guard let model = wire.model, !model.isEmpty else {
                return errorResponse(APIError(kind: .invalidRequest, message: "`model` is required.", code: "missing_model", param: "model"))
            }
            do {
                let result = try await manager.embed(EmbeddingRequest(inputs: wire.input.values, model: model))
                let data = try EmbeddingsResponse.json(result, model: model)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            } catch {
                return errorResponse(asAPIError(error))
            }
        }
    }

    /// Build the OpenAI `/v1/models` list body.
    static func modelsListJSON(_ ids: [String]) throws -> Data {
        let created = OpenAIID.timestamp()
        let items: [OAIJSON] = ids.map { id in
            .object([
                ("id", .string(id)),
                ("object", .string("model")),
                ("created", .int(created)),
                ("owned_by", .string("mlxz")),
            ])
        }
        let body: OAIJSON = .object([
            ("object", .string("list")),
            ("data", .array(items)),
        ])
        return try body.serialized()
    }

    /// The generic handler shared by every endpoint: decode → validate → translate →
    /// branch stream/non-stream. Adding an endpoint never touches this code.
    private func register<E: OpenAIEndpoint>(_ endpoint: E, on router: Router<BasicRequestContext>) {
        let manager = self.manager
        let gate = self.gate
        let logSink = self.logSink
        let metricsSink = self.metricsSink

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

            // Wait for a generation slot (FIFO) instead of rejecting — concurrent requests queue.
            // The limit is the current engine's `maxConcurrency`: 1 for single-sequence models (so
            // they serialize and the shared prefix cache holds), `maxBatch` for batchable models.
            // Only a bounded, full wait queue returns busy (genuine overload).
            guard await gate.acquire(limit: engine.maxConcurrency) else {
                return errorResponse(.busy())
            }

            logSink?("→ \(E.path) (\(genRequest.messages.count) messages, stream=\(endpoint.isStreaming(wire)))")

            if endpoint.isStreaming(wire) {
                return makeStreamingResponse(endpoint: endpoint, wire: wire, modelID: modelID, engine: engine, request: genRequest, gate: gate, metricsSink: metricsSink, logSink: logSink)
            } else {
                do {
                    let stream = try await engine.generate(genRequest)
                    let aggregated = try await AggregatedResult.collect(from: stream)
                    metricsSink?(aggregated.usage)
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
    gate: GenerationGate,
    metricsSink: (@Sendable (TokenUsage) -> Void)?,
    logSink: (@Sendable (String) -> Void)?
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
            // Release the generation slot exactly once, no matter how this closure exits
            // (normal finish, mid-stream error, or client-disconnect cancellation). A leaked
            // slot would wedge the depth-1 gate and reject all subsequent requests.
            let released = ReleaseOnce(gate: gate)
            // Live progress so generation isn't a black box: time-to-first-token (i.e. prefill
            // cost), a periodic tok/s heartbeat, and a final summary. Without this a long prefill
            // looks indistinguishable from a hang.
            let start = ContinuousClock.now
            var firstTokenAt: ContinuousClock.Instant? = nil
            var lastHeartbeat = start
            var tokenCount = 0
            do {
                let stream = try await engine.generate(request)
                for try await event in stream {
                    switch event {
                    case .completed(let result):
                        metricsSink?(result.usage)
                    case .textDelta:
                        tokenCount += 1
                        let now = ContinuousClock.now
                        if firstTokenAt == nil {
                            firstTokenAt = now
                            let ttft = Double((now - start).components.seconds)
                                + Double((now - start).components.attoseconds) / 1e18
                            logSink?(String(format: "  prefilled, first token in %.1fs", ttft))
                        } else if now - lastHeartbeat >= .seconds(1) {
                            lastHeartbeat = now
                            let dt = Double((now - firstTokenAt!).components.seconds)
                                + Double((now - firstTokenAt!).components.attoseconds) / 1e18
                            let tps = dt > 0 ? Double(tokenCount - 1) / dt : 0
                            logSink?(String(format: "  … %d tokens, %.1f tok/s", tokenCount, tps))
                        }
                    default:
                        break
                    }
                    for frame in try encoder.encode(event) {
                        try await writer.write(frame.byteBuffer(allocator: allocator))
                    }
                }
                for frame in encoder.terminator() {
                    try await writer.write(frame.byteBuffer(allocator: allocator))
                }
                let total = Double((ContinuousClock.now - start).components.seconds)
                    + Double((ContinuousClock.now - start).components.attoseconds) / 1e18
                let tps = total > 0 ? Double(tokenCount) / total : 0
                logSink?(String(format: "✓ done: %d tokens in %.1fs (%.1f tok/s)", tokenCount, total, tps))
                await released.release()
                try await writer.finish(nil)
            } catch {
                // Mid-stream errors or client disconnect: best-effort emit an error frame, finish.
                await released.release()
                let api = asAPIError(error)
                let errData = String(data: api.jsonBody(), encoding: .utf8) ?? "{}"
                let frame = SSEFrame(event: nil, data: errData)
                try? await writer.write(frame.byteBuffer(allocator: allocator))
                try? await writer.finish(nil)
            }
        }
    )
}

/// Releases a generation-gate slot at most once, even across error/cancellation paths.
private actor ReleaseOnce {
    private let gate: GenerationGate
    private var done = false
    init(gate: GenerationGate) { self.gate = gate }
    func release() async {
        guard !done else { return }
        done = true
        await gate.release()
    }
}

/// Build an OpenAI-format error response.
func errorResponse(_ error: APIError) -> Response {
    Response(
        status: error.httpStatus,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(data: error.jsonBody()))
    )
}
