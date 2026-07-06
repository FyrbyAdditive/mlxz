import Foundation
import MLX
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
    /// Fair scheduler for speculative requests — MTP or DSpark (interleaves decode steps so
    /// a short request doesn't wait for a long one). Built once, shared across requests.
    private let speculativeScheduler: SpeculativeScheduler
    /// The DSpark drafter attached at load (nil = no DSpark for this model).
    private let dspark: DSparkRuntimeBox?
    /// Fair scheduler for plain snapshot-reuse requests (rotating/hybrid-cache models — Gemma-3/4,
    /// gpt-oss). Interleaves prefill chunks and decode steps so a short request doesn't wait behind
    /// a whole long turn. Built once, shared across requests.
    private let plainScheduler: PlainScheduler
    /// Whether the loaded model conforms to `BatchableModel` (supports continuous batching).
    /// Captured once at load to avoid an async probe per request.
    private let isBatchable: Bool
    /// Whether plain prefix-cache requests go to the fair `PlainScheduler` instead of the
    /// serialized `cachedStream`. True for models whose caches CANNOT be soundly trimmed
    /// (rotating/hybrid — they use per-request snapshot copies, so sessions are isolated and can
    /// interleave). Trim-sound models share the mutable `box.cache` and must stay serialized at 1.
    /// Probed once at load (see `MLXModelLoader`).
    private let plainInterleaves: Bool

    /// `config.json` `model_type` (e.g. "gpt_oss", "qwen3_5"), if read at load. Refines output-format
    /// detection; nil is fine (repo id alone classifies the known families).
    private let modelType: String?
    /// The reasoning/tool-call output format this model uses, decided once at load.
    private let outputFormat: OutputFormat

    /// Concurrency limit the gate should apply for this model: serialize (1) the single-sequence
    /// shared-cache paths so the prefix cache holds; allow `maxBatch` only when the batch engine is
    /// actually used (batchable AND not running MTP). Computed once from load-time facts.
    public let maxConcurrency: Int

    public init(
        descriptor: ModelDescriptor,
        capabilities: ModelCapabilities,
        container: ModelContainer,
        perf: EnginePerfOptions = .default,
        isBatchable: Bool = false,
        modelType: String? = nil,
        trimUnsound: Bool = false,
        dspark: DSparkRuntimeBox? = nil
    ) {
        self.descriptor = descriptor
        self.capabilities = capabilities
        self.container = container
        self.perf = perf
        self.isBatchable = isBatchable
        self.modelType = modelType
        self.dspark = dspark
        self.outputFormat = OutputParserFactory.detectFormat(
            repoID: descriptor.repoID, modelType: modelType, capabilities: capabilities)
        self.promptCache = PromptCacheBox(
            prefixCacheSlots: perf.prefixCache ? perf.prefixCacheSlots : 0,
            prefixCacheBytesMB: perf.prefixCacheBytesMB)
        self.batchEngine = BatchGenerationEngine(container: container, maxBatch: perf.maxBatch)
        self.speculativeScheduler = SpeculativeScheduler(
            container: container, snapshotBlock: perf.prefixCache ? perf.snapshotBlock : 512,
            mode: dspark.map { .dspark($0) } ?? .mtp)
        self.plainScheduler = PlainScheduler(container: container)
        // Concurrency the gate admits. Speculative requests (MTP or DSpark) go to the fair
        // scheduler (which interleaves decode steps), batchable non-MTP requests go to the
        // batch engine, and non-trimmable (rotating/hybrid cache) plain models go to the fair
        // PlainScheduler — all want N requests admitted concurrently. A trim-sound plain
        // model (shared mutable box.cache) serializes at 1.
        let useMTPModel = perf.useMTP && capabilities.contains(.speculative)
        let plainInterleaves =
            trimUnsound && !useMTPModel && dspark == nil && !isBatchable && perf.prefixCache
        self.plainInterleaves = plainInterleaves
        self.maxConcurrency =
            (useMTPModel || dspark != nil || isBatchable || plainInterleaves)
            ? max(1, perf.maxBatch) : 1
    }

    /// Bench-only entry point: the speculative verify-cost curve (see `VerifyCurveBench`).
    public func runVerifyCurve(
        contexts: [Int], maxM: Int, itersPerPoint: Int
    ) async throws -> [VerifyCurveBench.Point] {
        try await VerifyCurveBench.run(
            container: container, perf: perf, repoID: descriptor.repoID,
            contexts: contexts, maxM: maxM, itersPerPoint: itersPerPoint)
    }

    public func generate(_ request: GenerationRequest) async throws -> AsyncThrowingStream<GenerationEvent, Error> {
        let parameters = Self.mapParameters(
            request.sampling, maxTokens: request.maxTokens, perf: perf, repoID: descriptor.repoID)
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
        let specDisabled = request.speculative?.mode == .disabled
        let mtpOptOut: Bool = {
            if case .draftModel = request.speculative?.mode { return true }
            return specDisabled
        }()
        // Image requests must take the vision-capable plain path: both the speculative scheduler and
        // the continuous-batching engine are TEXT-ONLY (they prepare `.text.tokens` and drop the image
        // pixel features), so routing an image request through either silently discards the image and
        // the model "sees" nothing. Keep them on `container.generate`, which runs the full VLM
        // prepare/merge (mergeInputIdsWithImageFeatures).
        let requestHasImages = request.messages.contains { $0.hasImages }
        let useMTP = perf.useMTP
            && dspark == nil
            && capabilities.contains(.speculative)
            && !mtpOptOut
            && !requestHasImages
        // DSpark draft-model speculation (standalone drafter attached at load). Greedy uses
        // exact-argmax verification; temperature > 0 uses speculative sampling (accept
        // w.p. min(1, p/q) + residual resample), which preserves the target distribution.
        let useDSpark = dspark != nil
            && !specDisabled
            && !requestHasImages

        // Capture only Sendable values (`request`, `container`, `parameters`). The non-Sendable
        // `UserInput`/`LMInput` are built and consumed entirely inside the task.
        return AsyncThrowingStream { continuation in
            let task = Task {
                continuation.yield(.started(.init(modelID: modelID)))
                // One streaming parser per request, selected for the model's output format
                // (`OutputParserFactory`). It splits the raw token stream into reasoning, visible text,
                // and tool calls — subsuming the old `ThinkParser`+`ToolCallParser` pair so all model
                // families (Qwen/Hermes, gpt-oss harmony, Mistral, Llama3, GLM4) parse correctly. The
                // default (`qwenHermes`) reproduces the previous behavior exactly.
                var sawToolCall = false  // true if ANY tool call (native or parsed from text) was emitted
                do {
                    let tools = request.tools.map { $0.map(Self.mapTool) }
                    let hasTools = (tools?.isEmpty == false)
                    // Only Qwen3.5/3.6 must disable thinking when tools are present: its template
                    // pre-opens a `<think>` block, and in an agentic flow the model narrates inside it
                    // and stops WITHOUT emitting the `<tool_call>`, so the agent stalls. Gemma (and
                    // others) are designed to reason AND call tools in the same turn, so thinking stays
                    // ON for them even with tools — otherwise reasoning never appears in agentic mode.
                    let thinkingDisabled = hasTools && outputFormat.disablesThinkingWithTools
                    let thinkingEnabled = !thinkingDisabled
                    // `enable_thinking` chat-template kwarg — only sent to formats that understand it
                    // (Qwen, Gemma). Gemma's template injects a `<|think|>` token when true, which makes
                    // it emit a separable reasoning channel; without it the reasoning is mixed into the
                    // answer. We send the explicit boolean so thinking is on by default and off when
                    // tools are present (so an agentic turn acts instead of musing).
                    let additionalContext: [String: any Sendable]? =
                        outputFormat.supportsEnableThinkingKwarg
                        ? ["enable_thinking": thinkingEnabled] : nil
                    // `startInsideThink` models Qwen's pre-opened `<think>` (the stream starts inside
                    // reasoning, only emits the closing tag). Only Qwen pre-opens; harmony etc. emit an
                    // explicit reasoning channel, so they start outside.
                    let startInsideThink =
                        outputFormat.prefersPreOpenedThink && !thinkingDisabled
                    var parser = OutputParserFactory.make(
                        format: outputFormat, startInsideThink: startInsideThink,
                        thinkingEnabled: thinkingEnabled)
                    // Reasoning-token budget (hard cap on the <think> block). Only meaningful when
                    // thinking is ON; per-request override else the engine default. 0 = uncapped.
                    let reasoningBudget = thinkingDisabled
                        ? 0
                        : max(0, request.reasoningTokenBudget ?? perf.reasoningTokenBudget ?? 0)

                    // If the conversation contains tool calls/results, replay it through the raw
                    // `.messages` dict path so `tool_calls`/`tool_call_id` reach the template
                    // (`Chat.Message` has no tool fields and would drop them, breaking the agent
                    // loop). Plain/image conversations keep the `.chat` path (which handles images).
                    let hasToolHistory = request.messages.contains {
                        !$0.toolCalls.isEmpty || $0.toolCallID != nil
                    }
                    let userInput: UserInput
                    if hasToolHistory {
                        // The messages-dict path doesn't extract images from the dicts, so collect
                        // them explicitly and pass via `images:` — otherwise an image sent in an
                        // agentic (tool-history) turn is silently dropped and the model "sees" nothing.
                        let images = request.messages.flatMap {
                            Self.images(from: $0, maxImagePixels: perf.maxImagePixels)
                        }
                        userInput = UserInput(
                            messages: request.messages.map(Self.mapMessageDict),
                            images: images,
                            tools: tools, additionalContext: additionalContext)
                    } else {
                        userInput = UserInput(
                            chat: request.messages.map { Self.mapMessage($0, maxImagePixels: perf.maxImagePixels) },
                            tools: tools, additionalContext: additionalContext)
                    }
                    let stream: AsyncStream<Generation>
                    if useMTP || useDSpark {
                        // Speculative decoding (MTP or DSpark) via the fair scheduler: requests
                        // interleave one decode step at a time, so a short request doesn't wait for a
                        // long one to finish (speculation can't batch, but it can be fair). Whole-
                        // prefix cache reuse via the LRU on the PromptCacheBox.
                        stream = await speculativeScheduler.submit(
                            userInput: userInput, parameters: parameters,
                            box: usePrefixCache ? promptCache : nil,
                            reasoningBudget: reasoningBudget)
                    } else if isBatchable && !requestHasImages {
                        // Continuous batching: concurrent plain requests decode together. Tokenize
                        // the prompt, then submit to the shared batch engine. (Image requests are
                        // excluded — the batch engine is text-only and would drop the pixels.)
                        let boxed = SendableValueBox(userInput)
                        let tokens = try await container.perform { context in
                            try await context.processor.prepare(input: boxed.consume())
                                .text.tokens.asArray(Int32.self)
                        }
                        let stopIds = await Self.stopTokenIds(container, format: outputFormat)
                        stream = await batchEngine.submit(
                            promptTokens: tokens,
                            maxTokens: parameters.maxTokens ?? 2048,
                            temperature: parameters.temperature,
                            stopTokenIds: stopIds)
                    } else if usePrefixCache && plainInterleaves && !requestHasImages {
                        // Fair interleaving for plain snapshot-reuse models (Gemma-3/4, gpt-oss):
                        // sessions advance one prefill chunk / one decode step per tick, so a short
                        // request (e.g. an IDE's title-gen) doesn't wait behind a whole long turn.
                        // Sessions resume from their own snapshot COPIES — no shared mutable cache.
                        // (Image requests keep the vision-capable plain path below.)
                        stream = await plainScheduler.submit(
                            userInput: userInput, parameters: parameters, box: promptCache)
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
                            let p = parser.consume(text)
                            if !p.reasoning.isEmpty {
                                continuation.yield(.reasoningDelta(p.reasoning))
                            }
                            if !p.visibleText.isEmpty {
                                continuation.yield(.textDelta(p.visibleText))
                            }
                            for call in p.toolCalls {
                                sawToolCall = true
                                continuation.yield(.toolCall(call))
                            }

                        case .toolCall(let nativeCall):
                            sawToolCall = true
                            continuation.yield(.toolCall(Self.mapNativeToolCall(nativeCall)))

                        case .info(let info):
                            let tail = parser.finish()
                            if !tail.reasoning.isEmpty {
                                continuation.yield(.reasoningDelta(tail.reasoning))
                            }
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
    /// For the harmony format, also include `<|return|>` (assistant EOS) and `<|call|>` (end of a tool
    /// call) so generation halts at a turn/tool boundary instead of running on past it.
    static func stopTokenIds(_ container: ModelContainer, format: OutputFormat = .qwenHermes) async -> Set<Int> {
        await container.perform { context in
            var ids = context.configuration.eosTokenIds
            if let t = context.tokenizer.eosTokenId { ids.insert(t) }
            for tok in context.configuration.extraEOSTokens {
                if let id = context.tokenizer.convertTokenToId(tok) { ids.insert(id) }
            }
            if format == .harmony {
                for tok in ["<|return|>", "<|call|>"] {
                    if let id = context.tokenizer.convertTokenToId(tok) { ids.insert(id) }
                }
            }
            return ids
        }
    }

    /// Whether every layer's cache can be SOUNDLY trimmed back to an arbitrary earlier position for
    /// prefix reuse. The fork's `canTrimPromptCache` returns true for a `RotatingKVCache` whenever its
    /// offset is still below the window, but that cache's `trim` only rewinds counters without
    /// restoring evicted ring-buffer entries — so trimming it corrupts state and the next prefill
    /// crashes. Hybrid sliding-window models (Gemma-3/4) mix rotating + full caches, so we must reject
    /// the whole set if ANY layer is rotating, and fall back to a fresh prefill.
    static func isSoundlyTrimmable(_ cache: [KVCache]) -> Bool {
        !cache.contains { $0 is RotatingKVCache }
    }

    /// Snapshot-reuse prefill for models whose caches can't be trimmed (rotating/hybrid). Looks up
    /// the longest LRU snapshot whose tokens exactly prefix the new prompt, resumes from a COPY of it
    /// (the LRU entry stays pristine), manually chunk-prefills up to the prompt-end boundary,
    /// snapshots that boundary for the next turn, and returns the cache + the small remaining suffix
    /// for the normal `generate` path (which prefills the suffix and steps the last token).
    ///
    /// Correctness: `RotatingKVCache.copy()`/`KVCacheSimple.copy()` deep-copy state, and resuming a
    /// copied prefix + feeding the rest is the same op sequence as a fresh prefill (causal attention,
    /// deterministic window eviction). Quantization is untouched: on this path the caches are always
    /// fp16 during prefill (maybeQuantizeKVCache runs later, inside TokenIterator's first step), so
    /// snapshots resume bit-identically to a fresh run.
    private static func snapshotPrefill(
        context: ModelContext,
        box: PromptCacheBox,
        fullInput: LMInput,
        newTokens: [Int32],
        parameters: GenerateParameters
    ) -> (cache: [KVCache], suffix: LMInput.Text) {
        let n = newTokens.count
        let step = parameters.prefillStepSize
        // Skip capturing when the prompt grew by less than this since the reused boundary — a
        // near-duplicate multi-GB snapshot buys almost nothing (re-prefilling <256 tokens is cheap).
        let captureGap = 256

        // Reuse: the longest snapshot that is an exact token-prefix of this prompt.
        let match = box.snapshotLRU.bestMatch(for: newTokens)
        let reused = match?.tokens.count ?? 0
        let cache = match.map { $0.modelCache.map { $0.copy() } }
            ?? context.model.newCache(parameters: parameters)

        if ProcessInfo.processInfo.environment["MLXZ_PREFIX_DIAG"] == "1" {
            let status = match != nil ? "HIT reused=\(reused)" : "MISS"
            FileHandle.standardError.write(
                Data("[PREFIX-SNAPSHOT] \(status) prompt=\(n) lruSlots=\(box.snapshotLRU.count)\n".utf8))
        }

        // Flatten to 1D: VLM processors (Gemma) emit [1, seq]; LLM processors emit [seq].
        let flat = fullInput.text.tokens.reshaped([n])

        // Capture at the prompt end minus one (the TokenIterator must step at least the last token
        // to produce the first logits). Agentic prompts grow append-only, so the prompt-end boundary
        // of THIS turn is exactly what the NEXT turn's prompt extends.
        let end = n - 1
        let capture = (end - reused >= captureGap && end >= 16) ? end : reused

        // Manually chunk-prefill [reused, capture), eval-ing between chunks to free each chunk's
        // transient graph (same discipline as the model's own chunked prepare). The chunk logits
        // are discarded UNEVALUATED — MLX's laziness means the lm_head projection never actually
        // runs for them (verified: skipping it explicitly benched 0% on a 24K prompt).
        var pos = reused
        while pos < capture {
            let take = min(step, capture - pos)
            let chunk = flat[pos ..< (pos + take)].expandedDimensions(axis: 0)
            _ = context.model(chunk, cache: cache)
            eval(cache)
            pos += take
        }

        // Snapshot the boundary (copies — `cache` keeps advancing through generation).
        if capture > reused {
            box.snapshotLRU.insert(
                tokens: Array(newTokens.prefix(capture)),
                modelCache: cache.map { $0.copy() },
                mtpCache: [])
        }

        // Remainder for the normal generate path (1D — model.prepare/TokenIterator batch it).
        let suffix = LMInput.Text(tokens: flat[capture...])
        return (cache, suffix)
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

            // Two reuse strategies, picked by what the model's caches support:
            //  - TRIM-reuse (plain caches, e.g. Qwen): trim the live cache back to the common prefix.
            //  - SNAPSHOT-reuse (rotating/hybrid caches, e.g. Gemma-3/4, gpt-oss): a `RotatingKVCache`
            //    reports `isTrimmable` while its offset is below the window, but its `trim` only
            //    decrements offset/idx without restoring dropped ring-buffer entries — trimming back
            //    to a prefix corrupts it and the next prefill crashes. `copy()` IS sound, so instead
            //    of trimming we snapshot (copy) the caches at the prompt boundary and, on the next
            //    request, resume from a copied snapshot whose tokens exactly prefix the new prompt.
            //    Without this, every agentic turn re-prefills the FULL history (~41s at 24K tokens).
            let freshProbe: [KVCache]? =
                box.cache == nil ? context.model.newCache(parameters: parameters) : nil
            let probeCache = box.cache ?? freshProbe!
            let trimSound = canTrimPromptCache(probeCache) && Self.isSoundlyTrimmable(probeCache)

            if !trimSound {
                // Snapshot-reuse path (rotating caches). Never touches box.cache/box.tokens — the
                // live-trim bookkeeping is meaningless for these models.
                let (kvCache, suffix) = Self.snapshotPrefill(
                    context: context, box: box, fullInput: fullInput,
                    newTokens: newTokens, parameters: parameters)
                return Self.withFullPromptCount(
                    try MLXLMCommon.generate(
                        input: LMInput(text: suffix),
                        cache: kvCache,
                        parameters: parameters,
                        context: context
                    ), promptTokenCount: newTokens.count)
            }

            let plan: PrefixCachePlan.Decision
            if box.cache != nil {
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
                // Fresh prefill: reuse the probe only if it IS fresh (no stale live cache).
                kvCache = freshProbe ?? context.model.newCache(parameters: parameters)
                inputForGeneration = fullInput
            }

            // Record what the cache will encode after this prompt is prefilled (prompt tokens).
            // Generated tokens extend the cache further; the next request reads the live offset.
            box.cache = kvCache
            box.tokens = newTokens

            return Self.withFullPromptCount(
                try MLXLMCommon.generate(
                    input: inputForGeneration,
                    cache: kvCache,
                    parameters: parameters,
                    context: context
                ), promptTokenCount: newTokens.count)
        }
    }

    /// Rewrite the final `.info` so `usage.prompt_tokens` reports the FULL prompt length. On the
    /// reuse paths, `generate()` only sees the un-cached suffix (it reported `prompt_tokens: 1` on
    /// a fully-reused 24K-token prompt), which breaks client-side context accounting.
    static func withFullPromptCount(
        _ inner: AsyncStream<Generation>, promptTokenCount: Int
    ) -> AsyncStream<Generation> {
        AsyncStream { continuation in
            let task = Task {
                for await item in inner {
                    if case .info(let info) = item {
                        continuation.yield(
                            .info(
                                GenerateCompletionInfo(
                                    promptTokenCount: promptTokenCount,
                                    generationTokenCount: info.generationTokenCount,
                                    promptTime: info.promptTime,
                                    generationTime: info.generateTime,
                                    stopReason: info.stopReason)))
                    } else {
                        continuation.yield(item)
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
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
