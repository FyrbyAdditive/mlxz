import Foundation
import MLXZCore

/// The fully-collected output of a generation, used to build a non-streaming response.
struct AggregatedResult: Sendable {
    var text: String = ""
    var toolCalls: [ToolCall] = []
    var finishReason: FinishReason = .stop
    var usage: TokenUsage = .init()

    /// Drain an event stream into a single aggregate.
    static func collect(from stream: AsyncThrowingStream<GenerationEvent, Error>) async throws -> AggregatedResult {
        var result = AggregatedResult()
        for try await event in stream {
            switch event {
            case .started:
                break
            case .textDelta(let t):
                result.text += t
            case .toolCall(let c):
                result.toolCalls.append(c)
            case .completed(let r):
                result.finishReason = r.finishReason
                result.usage = r.usage
            }
        }
        if !result.toolCalls.isEmpty {
            result.finishReason = .toolCalls
        }
        return result
    }
}

/// One OpenAI-compatible endpoint. Adding an endpoint = one conformer + one `register` call.
///
/// The wire request/response types and the streaming dialect are the only things that vary;
/// routing, validation, and the stream/non-stream branch are shared in `EndpointRegistry`.
protocol OpenAIEndpoint: Sendable {
    associatedtype WireRequest: Decodable & Sendable

    /// Route path, e.g. "/v1/chat/completions".
    static var path: String { get }

    /// Capabilities the loaded model must have to serve this endpoint.
    static var requiredCapabilities: ModelCapabilities { get }

    /// Whether the decoded request asked for a streamed response.
    func isStreaming(_ wire: WireRequest) -> Bool

    /// Translate the wire request into the engine-independent request.
    func toGenerationRequest(_ wire: WireRequest, modelID: String) throws -> GenerationRequest

    /// Build a non-streaming JSON response body (serialized text).
    func encodeNonStreaming(_ result: AggregatedResult, wire: WireRequest, modelID: String) throws -> Data

    /// Build the streaming encoder for this request.
    func makeStreamEncoder(for wire: WireRequest, modelID: String) -> any SSEEventEncoder
}
