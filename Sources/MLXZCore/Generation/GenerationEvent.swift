import Foundation

/// Why generation stopped.
public enum FinishReason: String, Sendable {
    case stop
    case length
    case toolCalls = "tool_calls"
    case cancelled
    case error
}

/// Token accounting and throughput for a completed generation.
public struct TokenUsage: Sendable, Equatable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var tokensPerSecond: Double?

    public var totalTokens: Int { promptTokens + completionTokens }

    public init(promptTokens: Int = 0, completionTokens: Int = 0, tokensPerSecond: Double? = nil) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.tokensPerSecond = tokensPerSecond
    }
}

/// Emitted once at the start of a generation.
public struct GenerationStart: Sendable {
    public var modelID: String
    public init(modelID: String) {
        self.modelID = modelID
    }
}

/// Emitted once when generation finishes.
public struct GenerationResult: Sendable {
    public var finishReason: FinishReason
    public var usage: TokenUsage
    public init(finishReason: FinishReason, usage: TokenUsage = .init()) {
        self.finishReason = finishReason
        self.usage = usage
    }
}

/// The single internal event stream, rich enough to drive both the chat-completions
/// delta dialect and the responses structured-event dialect.
public enum GenerationEvent: Sendable {
    case started(GenerationStart)
    case textDelta(String)
    /// A chunk of the model's chain-of-thought (`<think>…</think>`). Surfaced on its own channel so
    /// clients (e.g. VS Code) can render it as separate "thinking" rather than mixing it into the
    /// answer — and so it never leaks into the visible `content`.
    case reasoningDelta(String)
    case toolCall(ToolCall)
    case completed(GenerationResult)
}
