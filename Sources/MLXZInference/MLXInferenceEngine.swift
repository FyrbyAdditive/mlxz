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
                        // Native MTP self-speculative decoding, with whole-prefix cache reuse so a
                        // repeated system prompt (the VS Code case) is prefilled once, not per turn.
                        stream = try await Self.mtpStream(
                            container: container,
                            box: usePrefixCache ? promptCache : nil,
                            userInput: userInput, parameters: parameters)
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

    /// Build a generation stream using the model's native MTP head for self-speculative decoding.
    /// Runs inside `container.perform` because the model/caches are non-Sendable.
    static func mtpStream(
        container: ModelContainer,
        box: PromptCacheBox?,
        userInput: consuming UserInput,
        parameters: GenerateParameters
    ) async throws -> AsyncStream<Generation> {
        let boxedInput = SendableValueBox(userInput)
        return try await container.perform { context in
            let lmInput = try await context.processor.prepare(input: boxedInput.consume())
            let newTokens = lmInput.text.tokens.asArray(Int32.self)

            // Snapshot-and-restore prefix cache (SSM-safe). A snapshot encodes EXACTLY a stable
            // prefix (no generated tokens), so restoring it and prefilling only the new suffix is
            // sound even with the hybrid backbone's non-rewindable SSM layers. The recurring stable
            // prefix is a chat's constant system prompt; we snapshot it at the point it overlaps the
            // previous prompt and reuse it on later turns. The MTPCacheResult is the persistent
            // store: the SAME instance every request, so its snapshot survives across calls.
            let store = box?.mtpResult

            // 1) Restore: if a stored snapshot is a strict prefix of this prompt, reuse it.
            var restore: (model: [KVCache], mtp: [KVCache])? = nil
            var restoreCount = 0
            if let store, let m = store.snapshotModelCache, let h = store.snapshotMtpCache,
                let snapTokens = store.snapshotTokens
            {
                let n = MTPCacheReuse.reuseCount(snapshotTokens: snapTokens, newTokens: newTokens)
                if n > 0 {
                    restore = (m, h)
                    restoreCount = n
                }
            }

            // 2) Snapshot point: the prefix this prompt shares with the previous one (the stable
            // region likely to recur). Capture it during this prefill for the NEXT request.
            let snapshotAt = MTPCacheReuse.snapshotPoint(
                previousTokens: store?.promptTokens ?? [], currentTokens: newTokens)

            // Invalidate the stored snapshot up front: mtpGenerate refills `store` ONLY on clean
            // completion, so a cancelled/errored run leaves no stale snapshot for the next request.
            store?.snapshotModelCache = nil
            store?.snapshotMtpCache = nil
            store?.snapshotTokens = nil
            store?.promptTokens = nil

            return try mtpGenerate(
                input: lmInput, parameters: parameters, context: context,
                restore: restore, restoreCount: restoreCount, snapshotAt: snapshotAt, result: store)
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
