import Foundation

/// A loaded model that can generate completions. Knows nothing about Qwen, MLX, or OpenAI.
///
/// Implementations handle chat-template application, tokenization, sampling, and tool-call
/// parsing internally, exposing only the wire-independent `GenerationEvent` stream.
public protocol InferenceEngine: Sendable {
    /// The descriptor of the model this engine wraps.
    var descriptor: ModelDescriptor { get }

    /// What this model supports (drives endpoint advertisement and request validation).
    var capabilities: ModelCapabilities { get }

    /// Produce a stream of internal events for one request.
    ///
    /// Terminating iteration (e.g. an HTTP client disconnect) must cancel the underlying
    /// generation; conforming types wire this via the stream's `onTermination`.
    func generate(_ request: GenerationRequest) async throws -> AsyncThrowingStream<GenerationEvent, Error>
}
