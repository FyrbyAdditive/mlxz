import Foundation
import MLX
import MLXLMCommon
import MLXZCore

/// Fair, interleaving scheduler for PLAIN (non-MTP, non-batchable) models with snapshot-style
/// prefix reuse — the hybrid rotating-cache models (Gemma-3/4, gpt-oss). Without it those models
/// serialized WHOLE requests at the gate (`maxConcurrency == 1`), so a short utility request (an
/// IDE's title-generation) waited for an entire long agentic turn — up to ~40s behind a 24K-token
/// prefill. This scheduler advances every active session by one prefill CHUNK or one decode step
/// per tick, so a short request finishes in ~its own compute time alongside a long one.
///
/// Correctness: each session's math is identical to the old `cachedStream` snapshot path — the
/// same snapshot-LRU reuse + chunked prefill (`MLXInferenceEngine.snapshotPrefill` semantics), and
/// decode drives the fork's own `TokenIterator` (same sampler, repetition penalty, and
/// KV-quantization timing) with the same detokenizer + `ToolCallProcessor` handling as the fork's
/// `generate()` loop. Interleaving only reorders steps ACROSS sessions on the (container-
/// serialized) GPU; a session's own token sequence is unchanged. Sessions never share caches
/// (each resumes from its own snapshot COPY), so interleaving cannot clobber state.
actor PlainScheduler {
    private let container: ModelContainer

    /// A submission waiting to be built into a session inside the container.
    private struct Pending {
        let input: SendableValueBox<UserInput>
        let parameters: GenerateParameters
        let box: PromptCacheBox?
        let continuation: AsyncStream<Generation>.Continuation
        let id: Int
    }

    private struct Active {
        let session: PlainSession
        let id: Int
        let lru: SnapshotLRU?
    }

    private var pending: [Pending] = []
    private var cancelled: Set<Int> = []
    private var nextID = 0
    private var running = false

    init(container: ModelContainer) {
        self.container = container
    }

    /// Submit a plain request; returns its `Generation` stream. The scheduler starts on first submit.
    func submit(
        userInput: consuming UserInput,
        parameters: GenerateParameters,
        box: PromptCacheBox?
    ) -> AsyncStream<Generation> {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let id = nextID
        nextID += 1
        pending.append(
            Pending(
                input: SendableValueBox(userInput), parameters: parameters, box: box,
                continuation: continuation, id: id))
        continuation.onTermination = { [weak self] _ in
            Task { await self?.markCancelled(id) }
        }
        if !running {
            running = true
            Task { await self.run() }
        }
        return stream
    }

    private func markCancelled(_ id: Int) { cancelled.insert(id) }

    /// The scheduler loop, mirroring `MTPScheduler.run`: each tick performs one `container.perform`
    /// that admits new submissions and advances every active session. Snapshot inserts happen back
    /// on the actor between ticks.
    private func run() async {
        var active: [Active] = []

        while true {
            let toAdmit = pending
            pending = []
            let cancelledNow = cancelled

            if active.isEmpty && toAdmit.isEmpty {
                running = false
                return
            }

            // Fairness policy. Work units are wildly unequal — a 512-token prefill chunk costs
            // ~20× a decode step — so MTPScheduler's step-per-tick fairness would rate-limit a
            // short decoding request to ~1 token per long-prefill chunk (measured: no benefit at
            // all). Instead, decode-priority scheduling (the chunked-prefill policy from vLLM):
            // decoding sessions get a BURST of steps per tick, and prefilling sessions advance in
            // SMALLER chunks while contended, so a short request finishes in ~2-3× its solo time
            // while a long prefill still progresses every tick (no starvation).
            let contended = active.count + toAdmit.count > 1 || !cancelledNow.isEmpty
            let admitBox = SendableValueBox(toAdmit)
            let activeBox = SendableValueBox(active)
            let outBox = try? await container.perform { context in
                SendableValueBox(
                    await Self.tick(
                        context: context, admit: admitBox.consume(), active: activeBox.consume(),
                        cancelled: cancelledNow, contended: contended))
            }
            guard let outBox else { running = false; return }
            let out = outBox.consume()
            active = out.stillActive
            for ins in out.snapshots {
                ins.lru.insert(tokens: ins.tokens, modelCache: ins.model, mtpCache: ins.mtp)
            }
            await Task.yield()
        }
    }

    /// One tick (inside the container): admit new sessions, advance each active session by its
    /// phase-dependent budget, collect survivors + captured snapshots.
    private static func tick(
        context: ModelContext, admit: [Pending], active: [Active], cancelled: Set<Int>,
        contended: Bool
    ) async -> TickResult {
        var sessions = active

        for p in admit {
            if cancelled.contains(p.id) { p.continuation.finish(); continue }
            guard let lmInput = try? await context.processor.prepare(input: p.input.consume())
            else {
                p.continuation.finish()
                continue
            }
            let session = PlainSession(
                context: context, fullInput: lmInput, parameters: p.parameters,
                lru: p.box?.snapshotLRU, continuation: p.continuation)
            sessions.append(Active(session: session, id: p.id, lru: p.box?.snapshotLRU))
        }

        var stillActive: [Active] = []
        var snapshots: [SnapshotInsert] = []
        for entry in sessions {
            if cancelled.contains(entry.id) { entry.session.cancel(); continue }
            // Decode-priority budgets: decoding sessions run a burst of cheap steps per tick
            // (contended: enough to outweigh one small prefill chunk; solo: amortize tick
            // overhead). Prefilling sessions advance ONE chunk per tick — full-size solo, small
            // (128 tokens) when contended so decoders get the GPU back quickly. Chunk size does
            // not affect output bytes (verified 256–2048 identical on a 24K prompt).
            let budget = entry.session.phase == .decoding ? (contended ? 16 : 8) : 1
            var steps = 0
            while steps < budget, entry.session.phase != .finished {
                entry.session.stepOnce(context: context, contended: contended)
                if let snap = entry.session.takeCapturedSnapshot(), let lru = entry.lru {
                    snapshots.append(
                        SnapshotInsert(lru: lru, tokens: snap.tokens, model: snap.model, mtp: []))
                }
                steps += 1
            }
            if entry.session.phase != .finished { stillActive.append(entry) }
        }

        return TickResult(stillActive: stillActive, snapshots: snapshots)
    }

    private struct TickResult {
        let stillActive: [Active]
        let snapshots: [SnapshotInsert]
    }
}

/// One in-flight plain generation, steppable one prefill chunk / one decode token at a time.
/// `@unchecked Sendable`: only ever touched inside the serialized `container.perform` ticks (and
/// `cancel`, which only finishes the continuation). Mirrors the state machine of `MTPSession`.
final class PlainSession: @unchecked Sendable {
    enum Phase { case prefilling, decoding, finished }
    private(set) var phase: Phase = .prefilling

    private let parameters: GenerateParameters
    private let continuation: AsyncStream<Generation>.Continuation

    /// Full prompt, flattened 1-D, and its length (for prefill slicing and true usage reporting).
    private let flat: MLXArray
    private let promptTokens: [Int32]
    private let promptTokenCount: Int
    /// Prefill state: [reused, capture) is chunk-prefilled here; [capture...] goes to the iterator.
    private var pos: Int
    private let capture: Int
    private let reused: Int
    private var cache: [KVCache]

    /// Decode state (built at the prefill→decode transition).
    private var iterator: TokenIterator?
    private var detokenizer: NaiveStreamingDetokenizer
    private let toolCallProcessor: ToolCallProcessor
    private let stopTokenIds: Set<Int>
    private let unknownTokenId: Int?
    private var generated = 0

    /// Timing for the final `.info` (approximate wall times, matching what generate() reports).
    private let startTime = Date.timeIntervalSinceReferenceDate
    private var decodeStartTime: TimeInterval = 0

    /// Snapshot captured at the boundary, handed to the actor for LRU insertion between ticks.
    private var capturedSnapshot: (tokens: [Int32], model: [KVCache])?

    init(
        context: ModelContext, fullInput: LMInput, parameters: GenerateParameters,
        lru: SnapshotLRU?, continuation: AsyncStream<Generation>.Continuation
    ) {
        self.parameters = parameters
        self.continuation = continuation

        let newTokens = fullInput.text.tokens.asArray(Int32.self)
        let n = newTokens.count
        self.promptTokens = newTokens
        self.promptTokenCount = n
        // Flatten to 1D: VLM processors (Gemma) emit [1, seq]; LLM processors emit [seq].
        self.flat = fullInput.text.tokens.reshaped([n])

        // Reuse: the longest snapshot that is an exact token-prefix of this prompt (same policy
        // and captureGap as `MLXInferenceEngine.snapshotPrefill`).
        let captureGap = 256
        let match = lru?.bestMatch(for: newTokens)
        let reused = match?.tokens.count ?? 0
        self.reused = reused
        self.pos = reused
        self.cache = match.map { $0.modelCache.map { $0.copy() } }
            ?? context.model.newCache(parameters: parameters)
        let end = n - 1
        self.capture = (end - reused >= captureGap && end >= 16) ? end : reused

        if ProcessInfo.processInfo.environment["MLXZ_PREFIX_DIAG"] == "1" {
            let status = match != nil ? "HIT reused=\(reused)" : "MISS"
            FileHandle.standardError.write(
                Data("[PREFIX-SNAPSHOT] \(status) prompt=\(n) (interleaved)\n".utf8))
        }

        // Same stop set as the fork's generate() loop (`buildStopTokenIds`): model EOS ids +
        // tokenizer EOS + extra EOS strings. (No format extras — identical to the previous path.)
        var stops = context.configuration.eosTokenIds
        if let t = context.tokenizer.eosTokenId { stops.insert(t) }
        for tok in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(tok) { stops.insert(id) }
        }
        self.stopTokenIds = stops
        self.unknownTokenId = context.tokenizer.unknownTokenId
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
        self.toolCallProcessor = ToolCallProcessor(
            format: context.configuration.toolCallFormat ?? .json)
    }

    /// Advance by one unit of work: one prefill chunk, or one decode token.
    func stepOnce(context: ModelContext, contended: Bool) {
        switch phase {
        case .prefilling: prefillStepOnce(context: context, contended: contended)
        case .decoding: decodeStepOnce()
        case .finished: break
        }
    }

    func takeCapturedSnapshot() -> (tokens: [Int32], model: [KVCache])? {
        defer { capturedSnapshot = nil }
        return capturedSnapshot
    }

    func cancel() {
        phase = .finished
        continuation.finish()
    }

    /// One prefill chunk (the same loop body as `snapshotPrefill`, one iteration per tick). On
    /// reaching the capture boundary: snapshot it for the next turn, then build the TokenIterator
    /// over the remaining suffix (≤ captureGap+1 tokens — bounded admission cost) and switch to
    /// decoding.
    private func prefillStepOnce(context: ModelContext, contended: Bool) {
        if pos < capture {
            // Contended: prefill in small chunks so decoding sessions get the GPU back within
            // ~a decode-burst's time. Solo: full-size chunks (fastest total prefill).
            let step = contended
                ? min(parameters.prefillStepSize, 128) : parameters.prefillStepSize
            let take = min(step, capture - pos)
            let chunk = flat[pos ..< (pos + take)].expandedDimensions(axis: 0)
            _ = context.model(chunk, cache: cache)
            eval(cache)
            pos += take
            if pos < capture { return }
        }
        // Boundary reached: capture the snapshot (copies — `cache` keeps advancing through
        // generation), then hand the small suffix to a TokenIterator, exactly as the old path
        // handed it to `generate(input: suffix, cache:)`.
        if capture > reused {
            capturedSnapshot = (
                tokens: Array(promptTokens.prefix(capture)),
                model: cache.map { $0.copy() }
            )
        }
        let suffix = LMInput.Text(tokens: flat[capture...])
        do {
            iterator = try TokenIterator(
                input: LMInput(text: suffix), model: context.model, cache: cache,
                parameters: parameters)
        } catch {
            finish(reason: .cancelled)
            return
        }
        decodeStartTime = Date.timeIntervalSinceReferenceDate
        phase = .decoding
    }

    /// One decode token, replicating the fork's `generateLoopTask` body: stop-token check, then
    /// detokenize → tool-call processor → emit text/tool events.
    private func decodeStepOnce() {
        guard var it = iterator else {
            finish(reason: .cancelled)
            return
        }
        guard let token = it.next() else {
            iterator = it
            // Iterator exhausted: maxTokens reached (or nothing to step).
            finish(reason: .length)
            return
        }
        iterator = it

        if token == unknownTokenId || stopTokenIds.contains(token) {
            finish(reason: .stop)
            return
        }

        generated += 1
        detokenizer.append(token: token)
        if let chunk = detokenizer.next() {
            if let textToYield = toolCallProcessor.processChunk(chunk) {
                continuation.yield(.chunk(textToYield))
            }
            if let toolCall = toolCallProcessor.toolCalls.popLast() {
                continuation.yield(.toolCall(toolCall))
            }
        }
    }

    private func finish(reason: GenerateStopReason) {
        // Same end-of-generation handling as the fork's TextToolTokenLoopHandler.
        toolCallProcessor.processEOS()
        for toolCall in toolCallProcessor.toolCalls {
            continuation.yield(.toolCall(toolCall))
        }
        let now = Date.timeIntervalSinceReferenceDate
        let promptTime = decodeStartTime > 0 ? decodeStartTime - startTime : now - startTime
        let generationTime = decodeStartTime > 0 ? now - decodeStartTime : 0
        continuation.yield(
            .info(
                GenerateCompletionInfo(
                    promptTokenCount: promptTokenCount,
                    generationTokenCount: generated,
                    promptTime: promptTime,
                    generationTime: generationTime,
                    stopReason: reason)))
        phase = .finished
        continuation.finish()
    }
}
