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

    /// How many generations may run concurrently for this model. `1` serializes them — required for
    /// the single-sequence paths (MTP / prefix-cached) that share one prefix cache, so concurrent
    /// requests can't clobber each other's cached prefix (they queue at the gate instead). Batchable
    /// models that use the continuous-batching engine return `maxBatch` so requests decode together.
    var maxConcurrency: Int { get }

    /// Human-readable description of the active speculative-decoding mode, if any —
    /// e.g. "DSpark drafter: deepseek-ai/dspark_qwen3_8b_block7" or "native MTP".
    /// nil = plain decoding. Surfaced in the UI/logs so users can see speculation is on.
    var speculationStatus: String? { get }

    /// Produce a stream of internal events for one request.
    ///
    /// Terminating iteration (e.g. an HTTP client disconnect) must cancel the underlying
    /// generation; conforming types wire this via the stream's `onTermination`.
    func generate(_ request: GenerationRequest) async throws -> AsyncThrowingStream<GenerationEvent, Error>
}

extension InferenceEngine {
    /// Default: serialize. Engines that support continuous batching override this.
    public var maxConcurrency: Int { 1 }

    /// Default: plain decoding.
    public var speculationStatus: String? { nil }
}
