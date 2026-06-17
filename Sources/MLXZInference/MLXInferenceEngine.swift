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
    /// Persistent prefix cache, reused across requests on the single-sequence paths (MTP /
    /// cachedStream). Mutated only inside `container.perform`; the GenerationGate serializes those
    /// paths (`maxConcurrency == 1`), so concurrent requests can't clobber each other's cached
    /// prefix — they queue. (Batchable models don't touch this box.)
    private let promptCache: PromptCacheBox
    private let perf: EnginePerfOptions
    /// Continuous-batching engine for plain (non-MTP) requests, so concurrent requests decode
    /// together instead of being serialized/rejected. Built once, shared across requests.
    private let batchEngine: BatchGenerationEngine
    /// Fair scheduler for MTP requests (interleaves decode steps so a short request doesn't wait for
    /// a long one). Built once, shared across requests.
    private let mtpScheduler: MTPScheduler
    /// Whether the loaded model conforms to `BatchableModel` (supports continuous batching).
    /// Captured once at load to avoid an async probe per request.
    private let isBatchable: Bool

    /// Concurrency limit the gate should apply for this model: serialize (1) the single-sequence
    /// shared-cache paths so the prefix cache holds; allow `maxBatch` only when the batch engine is
    /// actually used (batchable AND not running MTP). Computed once from load-time facts.
    public let maxConcurrency: Int

    public init(
        descriptor: ModelDescriptor,
        capabilities: ModelCapabilities,
        container: ModelContainer,
        perf: EnginePerfOptions = .default,
        isBatchable: Bool = false
    ) {
        self.descriptor = descriptor
        self.capabilities = capabilities
        self.container = container
        self.perf = perf
        self.isBatchable = isBatchable
        self.promptCache = PromptCacheBox(
            prefixCacheSlots: perf.prefixCache ? perf.prefixCacheSlots : 0,
            prefixCacheBytesMB: perf.prefixCacheBytesMB)
        self.batchEngine = BatchGenerationEngine(container: container, maxBatch: perf.maxBatch)
        self.mtpScheduler = MTPScheduler(
            container: container, snapshotBlock: perf.prefixCache ? perf.snapshotBlock : 512)
        // Concurrency the gate admits. MTP requests go to the fair scheduler (which interleaves
        // decode steps), and batchable non-MTP requests go to the batch engine — both want N
        // requests admitted concurrently. A plain non-batchable model still serializes at 1.
        let useMTPModel = perf.useMTP && capabilities.contains(.speculative)
        self.maxConcurrency = (useMTPModel || isBatchable) ? max(1, perf.maxBatch) : 1
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

        // Use native MTP self-speculative decoding whenever the model has an MTP head (it is a
        // pure speedup with identical output), unless the request explicitly opts out by setting
        // a non-MTP speculative mode. Text-only requests without a fixed seed.
        let mtpOptOut: Bool = {
            if case .draftModel = request.speculative?.mode { return true }
            return false
        }()
        let useMTP = perf.useMTP
            && capabilities.contains(.speculative)
            && !mtpOptOut
            && !request.messages.contains { $0.hasImages }

        // Capture only Sendable values (`request`, `container`, `parameters`). The non-Sendable
        // `UserInput`/`LMInput` are built and consumed entirely inside the task.
        return AsyncThrowingStream { continuation in
            let task = Task {
                continuation.yield(.started(.init(modelID: modelID)))
                // Split the model's <think>…</think> chain-of-thought out of the text BEFORE
                // tool-call parsing, so reasoning is surfaced on its own channel (clients render it
                // separately) and never leaks into the visible content or gets mis-parsed as a tool
                // call. Visible text then flows through the tool-call fallback parser as before.
                var think = ThinkParser()
                var fallback = ToolCallParser()
                var sawToolCall = false  // true if ANY tool call (native or parsed from text) was emitted
                do {
                    let tools = request.tools.map { $0.map(Self.mapTool) }
                    // Qwen3.5/3.6's chat template pre-opens a `<think>` block on every assistant
                    // turn. In an agentic flow the model then narrates its plan inside that block
                    // and stops at <|im_end|> WITHOUT ever closing `</think>` or emitting the
                    // `<tool_call>` — so the VS Code agent gets no tool call and stalls (and the
                    // reasoning leaks). When the request carries tools, disable thinking
                    // (`enable_thinking: false` → the template emits an empty `<think></think>` and
                    // goes straight to the answer/tool-call) so the model acts instead of musing.
                    let additionalContext: [String: any Sendable]? =
                        (tools?.isEmpty == false) ? ["enable_thinking": false] : nil

                    // If the conversation contains tool calls/results, replay it through the raw
                    // `.messages` dict path so `tool_calls`/`tool_call_id` reach the template
                    // (`Chat.Message` has no tool fields and would drop them, breaking the agent
                    // loop). Plain/image conversations keep the `.chat` path (which handles images).
                    let hasToolHistory = request.messages.contains {
                        !$0.toolCalls.isEmpty || $0.toolCallID != nil
                    }
                    let userInput: UserInput
                    if hasToolHistory {
                        userInput = UserInput(
                            messages: request.messages.map(Self.mapMessageDict),
                            tools: tools, additionalContext: additionalContext)
                    } else {
                        userInput = UserInput(
                            chat: request.messages.map(Self.mapMessage),
                            tools: tools, additionalContext: additionalContext)
                    }
                    let stream: AsyncStream<Generation>
                    if useMTP {
                        // Native MTP self-speculative decoding via the fair scheduler: requests
                        // interleave one decode step at a time, so a short request doesn't wait for a
                        // long one to finish (MTP can't batch, but it can be fair). Whole-prefix cache
                        // reuse via the LRU on the PromptCacheBox.
                        stream = await mtpScheduler.submit(
                            userInput: userInput, parameters: parameters,
                            box: usePrefixCache ? promptCache : nil)
                    } else if isBatchable {
                        // Continuous batching: concurrent plain requests decode together. Tokenize
                        // the prompt, then submit to the shared batch engine.
                        let boxed = SendableValueBox(userInput)
                        let tokens = try await container.perform { context in
                            try await context.processor.prepare(input: boxed.consume())
                                .text.tokens.asArray(Int32.self)
                        }
                        let stopIds = await Self.stopTokenIds(container)
                        stream = await batchEngine.submit(
                            promptTokens: tokens,
                            maxTokens: parameters.maxTokens ?? 2048,
                            temperature: parameters.temperature,
                            stopTokenIds: stopIds)
                    } else if usePrefixCache {
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
                            let split = think.consume(text)
                            if !split.reasoning.isEmpty {
                                continuation.yield(.reasoningDelta(split.reasoning))
                            }
                            if !split.visibleText.isEmpty {
                                let parsed = fallback.consume(split.visibleText)
                                if !parsed.visibleText.isEmpty {
                                    continuation.yield(.textDelta(parsed.visibleText))
                                }
                                for call in parsed.toolCalls {
                                    sawToolCall = true
                                    continuation.yield(.toolCall(call))
                                }
                            }

                        case .toolCall(let nativeCall):
                            sawToolCall = true
                            continuation.yield(.toolCall(Self.mapNativeToolCall(nativeCall)))

                        case .info(let info):
                            let thinkTail = think.finish()
                            if !thinkTail.reasoning.isEmpty {
                                continuation.yield(.reasoningDelta(thinkTail.reasoning))
                            }
                            var tail = fallback.consume(thinkTail.visibleText)
                            let flushed = fallback.finish()
                            tail.visibleText += flushed.visibleText
                            tail.toolCalls += flushed.toolCalls
                            if !tail.visibleText.isEmpty {
                                continuation.yield(.textDelta(tail.visibleText))
                            }
                            for call in tail.toolCalls {
                                sawToolCall = true
                                continuation.yield(.toolCall(call))
                            }
                            let usage = TokenUsage(
                                promptTokens: info.promptTokenCount,
                                completionTokens: info.generationTokenCount,
                                tokensPerSecond: info.tokensPerSecond
                            )
                            let reason = Self.mapStopReason(info.stopReason, sawToolCall: sawToolCall)
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

    /// The stop-token id set (model EOS + tokenizer EOS + extra EOS strings) for the batch engine.
    static func stopTokenIds(_ container: ModelContainer) async -> Set<Int> {
        await container.perform { context in
            var ids = context.configuration.eosTokenIds
            if let t = context.tokenizer.eosTokenId { ids.insert(t) }
            for tok in context.configuration.extraEOSTokens {
                if let id = context.tokenizer.convertTokenToId(tok) { ids.insert(id) }
            }
            return ids
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
/// Internal (not private) so `BatchGenerationEngine` in the same module can reuse it.
final class SendableValueBox<T>: @unchecked Sendable {
    private var value: T?
    init(_ value: T) { self.value = value }
    func consume() -> T {
        defer { value = nil }
        return value!
    }
}
