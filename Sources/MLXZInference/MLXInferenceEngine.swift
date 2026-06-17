import Foundation
import MLXZCore
import MLXLMCommon

/// An `InferenceEngine` backed by an mlx-swift-lm `ModelContainer`. This is the only type
/// that touches MLX. It maps our wire-independent `GenerationRequest` onto the package's
/// stateless `generate(input:parameters:)` path and translates `Generation` events back.
///
/// Speculative decoding (`request.speculative`): the seam is present but inert. mlx-swift-lm
/// 3.31.3 has no native single-model MTP, and its draft-model path (`generate(...draftModel:)`)
/// needs the *main* `ModelContext` and the *draft* `any LanguageModel` in one isolation domain —
/// but `any LanguageModel` is non-Sendable and each model lives in its own `ModelContainer` actor,
/// so a draft model cannot be moved into the main container's `perform`. Cleanly supporting it
/// requires upstream to host both models in one container or expose a Sendable speculative entry
/// point. Until then speculative requests use standard decoding (correct output, no speedup);
/// `.speculative` is still advertised so the UI/Copilot can surface MTP models.
public struct MLXInferenceEngine: InferenceEngine {
    public let descriptor: ModelDescriptor
    public let capabilities: ModelCapabilities

    private let container: ModelContainer
    /// Persistent prefix cache, reused across requests. Mutated only inside `container.perform`,
    /// where the GenerationGate already serializes access.
    private let promptCache = PromptCacheBox()
    private let perf: EnginePerfOptions

    public init(
        descriptor: ModelDescriptor,
        capabilities: ModelCapabilities,
        container: ModelContainer,
        perf: EnginePerfOptions = .default
    ) {
        self.descriptor = descriptor
        self.capabilities = capabilities
        self.container = container
        self.perf = perf
    }

    public func generate(_ request: GenerationRequest) async throws -> AsyncThrowingStream<GenerationEvent, Error> {
        let parameters = Self.mapParameters(request.sampling, maxTokens: request.maxTokens, perf: perf)
        let modelID = descriptor.repoID
        let container = self.container
        let promptCache = self.promptCache
        // Reuse the prefix cache only for plain text requests (images complicate token alignment,
        // and a fixed seed implies the caller wants fully deterministic, fresh decoding).
        // A bounded (rotating) or non-trimmable cache disables reuse via canTrimPromptCache.
        let usePrefixCache = perf.prefixCache
            && !request.messages.contains { $0.hasImages }
            && request.sampling.seed == nil

        // Capture only Sendable values (`request`, `container`, `parameters`). The non-Sendable
        // `UserInput`/`LMInput` are built and consumed entirely inside the task.
        return AsyncThrowingStream { continuation in
            let task = Task {
                continuation.yield(.started(.init(modelID: modelID)))
                // Fallback parser for models whose tool calls arrive as raw <tool_call> text
                // rather than as native `.toolCall` events.
                var fallback = ToolCallParser()
                var sawNativeToolCall = false
                do {
                    let chat = request.messages.map(Self.mapMessage)
                    let tools = request.tools.map { $0.map(Self.mapTool) }
                    let userInput = UserInput(chat: chat, tools: tools)
                    let stream: AsyncStream<Generation>
                    if usePrefixCache {
                        stream = try await Self.cachedStream(
                            container: container, box: promptCache,
                            userInput: userInput, parameters: parameters)
                    } else {
                        let lmInput = try await container.prepare(input: userInput)
                        stream = try await container.generate(input: lmInput, parameters: parameters)
                    }
                    for await item in stream {
                        if Task.isCancelled { break }
                        switch item {
                        case .chunk(let text):
                            let parsed = fallback.consume(text)
                            if !parsed.visibleText.isEmpty {
                                continuation.yield(.textDelta(parsed.visibleText))
                            }
                            for call in parsed.toolCalls {
                                continuation.yield(.toolCall(call))
                            }

                        case .toolCall(let nativeCall):
                            sawNativeToolCall = true
                            continuation.yield(.toolCall(Self.mapNativeToolCall(nativeCall)))

                        case .info(let info):
                            let tail = fallback.finish()
                            if !tail.visibleText.isEmpty {
                                continuation.yield(.textDelta(tail.visibleText))
                            }
                            for call in tail.toolCalls {
                                continuation.yield(.toolCall(call))
                            }
                            let usage = TokenUsage(
                                promptTokens: info.promptTokenCount,
                                completionTokens: info.generationTokenCount,
                                tokensPerSecond: info.tokensPerSecond
                            )
                            let reason = Self.mapStopReason(info.stopReason, sawToolCall: sawNativeToolCall)
                            continuation.yield(.completed(.init(finishReason: reason, usage: usage)))

                        @unknown default:
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Build a generation stream that reuses a persistent KV cache for the shared prompt prefix.
    ///
    /// Runs inside `container.perform` because the cache, model, and iterator are non-Sendable.
    /// Algorithm: tokenize the full prompt, find the longest common prefix with what the cache
    /// already encodes, trim the cache back to that prefix, and feed only the new suffix tokens —
    /// the model then attends to the cached prefix instead of re-prefilling it.
    static func cachedStream(
        container: ModelContainer,
        box: PromptCacheBox,
        userInput: consuming UserInput,
        parameters: GenerateParameters
    ) async throws -> AsyncStream<Generation> {
        let boxedInput = SendableValueBox(userInput)
        return try await container.perform { context in
            let fullInput = try await context.processor.prepare(input: boxedInput.consume())
            let newTokens = fullInput.text.tokens.asArray(Int32.self)

            // Decide how much of the existing cache to reuse.
            let plan: PrefixCachePlan.Decision
            if let cache = box.cache, canTrimPromptCache(cache) {
                plan = PrefixCachePlan.plan(cachedTokens: box.tokens, newTokens: newTokens)
            } else {
                plan = .fresh
            }

            let kvCache: [KVCache]
            let inputForGeneration: LMInput

            if plan.reuse, let cache = box.cache {
                // Trim the cache down to the common prefix, then feed only the suffix.
                let liveOffset = cache.first?.offset ?? 0
                let toTrim = liveOffset - plan.reuseCount
                if toTrim > 0 { trimPromptCache(cache, numTokens: toTrim) }
                kvCache = cache
                let suffix = fullInput.text[text: plan.reuseCount...]
                inputForGeneration = LMInput(text: suffix)
            } else {
                kvCache = context.model.newCache(parameters: parameters)
                inputForGeneration = fullInput
            }

            // Record what the cache will encode after this prompt is prefilled (prompt tokens).
            // Generated tokens extend the cache further; the next request reads the live offset.
            box.cache = kvCache
            box.tokens = newTokens

            return try MLXLMCommon.generate(
                input: inputForGeneration,
                cache: kvCache,
                parameters: parameters,
                context: context
            )
        }
    }
}

/// Minimal box to move a non-Sendable value into a `@Sendable` closure exactly once.
private final class SendableValueBox<T>: @unchecked Sendable {
    private var value: T?
    init(_ value: T) { self.value = value }
    func consume() -> T {
        defer { value = nil }
        return value!
    }
}
