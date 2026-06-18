import Foundation

/// A request to generate a completion, fully independent of OpenAI wire shapes.
/// Endpoints translate their wire requests into this; engines consume it.
public struct GenerationRequest: Sendable {
    public var messages: [ChatMessage]
    public var sampling: SamplingParameters
    public var maxTokens: Int?
    public var stop: [String]
    public var tools: [ToolDefinition]?
    public var speculative: SpeculativeConfig?
    /// Max tokens the model may spend inside a `<think>` reasoning block before it is force-closed
    /// and the model must answer. nil = use the engine default; ≤0 = uncapped. Set per request from
    /// `reasoning_effort`/`max_reasoning_tokens` (else the engine's configured default applies).
    public var reasoningTokenBudget: Int?
    /// Stable id used for logging and cancellation correlation.
    public var requestID: String

    public init(
        messages: [ChatMessage],
        sampling: SamplingParameters = .default,
        maxTokens: Int? = nil,
        stop: [String] = [],
        tools: [ToolDefinition]? = nil,
        speculative: SpeculativeConfig? = nil,
        reasoningTokenBudget: Int? = nil,
        requestID: String = UUID().uuidString
    ) {
        self.messages = messages
        self.sampling = sampling
        self.maxTokens = maxTokens
        self.stop = stop
        self.tools = tools
        self.speculative = speculative
        self.reasoningTokenBudget = reasoningTokenBudget
        self.requestID = requestID
    }
}
