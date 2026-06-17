import Foundation
import MLX
import MLXLMCommon
import MLXZCore

/// Continuous-batching generation engine: multiple requests decode together in one batched model
/// forward pass per step, so concurrent Copilot/API requests run concurrently instead of being
/// rejected (the `server_busy` 429) or serialized one-at-a-time.
///
/// Design (mirrors mlx-lm's `BatchGenerator`):
/// - Requests `submit(...)` and receive an `AsyncStream<Generation>`; a single scheduler task drives
///   all of them. The scheduler holds the model's `ModelContainer`, so each batched forward runs
///   under the container's serial lock — one forward at a time, but N sequences per forward.
/// - Each step decodes the running batch (B,1) and samples per-row; finished rows are filtered out
///   (the batch shrinks); waiting newcomers are prefilled and `extend`-ed into the running batch.
/// - The model must conform to `BatchableModel` (text Qwen3.5 + VLM Qwen3.5 do). MTP/speculative
///   requests are NOT handled here — the caller routes them to the single-sequence path.
///
/// Correctness rests on the proven batched caches: `BatchKVCache` (per-row attention offsets +
/// left-padding mask) and batched `MambaCache` (left-padding-masked GatedDeltaNet recurrence).
public actor BatchGenerationEngine {
    private let container: ModelContainer
    private let maxBatch: Int

    /// One in-flight sequence in the batch. `@unchecked Sendable`: only ever read/mutated inside the
    /// serialized `container.perform` closure (one batch step at a time) or the owning actor.
    final class Seq: @unchecked Sendable {
        let id: Int
        let promptTokens: [Int32]
        let maxTokens: Int
        let temperature: Float
        let stopTokenIds: Set<Int>
        let continuation: AsyncStream<Generation>.Continuation
        /// Created lazily inside the scheduler (needs the tokenizer from the ModelContainer).
        var detokenizer: NaiveStreamingDetokenizer?
        var generated = 0
        var lastToken: Int32 = 0
        var finished = false
        init(
            id: Int, promptTokens: [Int32], maxTokens: Int, temperature: Float,
            stopTokenIds: Set<Int>, continuation: AsyncStream<Generation>.Continuation
        ) {
            self.id = id
            self.promptTokens = promptTokens
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.stopTokenIds = stopTokenIds
            self.continuation = continuation
        }
    }

    private var waiting: [Seq] = []
    private var nextID = 0
    private var schedulerRunning = false

    public init(container: ModelContainer, maxBatch: Int = 8) {
        self.container = container
        self.maxBatch = max(1, maxBatch)
    }

    /// Submit a prompt (already tokenized 1-D) for batched generation. Returns a stream of
    /// `Generation` events (`.chunk` text + a final `.info`). The scheduler starts on first submit.
    public func submit(
        promptTokens: [Int32], maxTokens: Int, temperature: Float, stopTokenIds: Set<Int>
    ) -> AsyncStream<Generation> {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let id = nextID
        nextID += 1
        let seq = Seq(
            id: id, promptTokens: promptTokens, maxTokens: maxTokens, temperature: temperature,
            stopTokenIds: stopTokenIds, continuation: continuation)
        waiting.append(seq)
        continuation.onTermination = { [weak self] _ in
            Task { await self?.cancel(id: id) }
        }
        if !schedulerRunning {
            schedulerRunning = true
            Task { await self.runScheduler() }
        }
        return stream
    }

    private func cancel(id: Int) {
        // Mark a waiting/running sequence cancelled; the scheduler drops it on the next step.
        if let s = waiting.first(where: { $0.id == id }) { s.finished = true }
        cancelled.insert(id)
    }

    private var cancelled: Set<Int> = []

    /// The scheduler loop: runs until no sequences remain. Each iteration performs ONE batched
    /// forward inside `container.perform` (which serializes GPU access), advancing all live rows.
    private func runScheduler() async {
        // Drain until there is genuinely nothing left.
        while true {
            let batch = takeReadyBatch()
            if batch.isEmpty { schedulerRunning = false; return }
            await runBatch(batch)
            // Loop: newly-submitted requests (arrived during runBatch) are picked up next round.
            if waiting.isEmpty { schedulerRunning = false; return }
        }
    }

    /// Pull up to `maxBatch` waiting sequences to form the next batch (v1: static per-batch; a
    /// batch runs to completion of all its rows before the next batch starts — newcomers that
    /// arrive mid-batch wait for the next round). Skips cancelled sequences.
    private func takeReadyBatch() -> [Seq] {
        var batch: [Seq] = []
        while !waiting.isEmpty && batch.count < maxBatch {
            let s = waiting.removeFirst()
            if cancelled.contains(s.id) { s.continuation.finish(); continue }
            batch.append(s)
        }
        return batch
    }

    /// Prefill + decode a batch of sequences to completion, demuxing tokens to each stream. The
    /// heavy lifting runs in a free function inside `container.perform` (the serial GPU lock); the
    /// cancelled-set is snapshotted in so the `@Sendable` closure touches no actor state.
    private func runBatch(_ batch: [Seq]) async {
        let cancelledSnapshot = cancelled
        let boxed = SendableValueBox(batch)
        await container.perform { context in
            runBatchStep(batch: boxed.consume(), context: context, cancelled: cancelledSnapshot)
        }
    }
}

/// One batch run to completion: prefill, then decode step-by-step with per-row sampling, stop
/// detection, stream demux, and cache `filter` as rows finish. Free function so it can run inside
/// the `@Sendable` `container.perform` closure without actor-isolation conflicts.
private func runBatchStep(
    batch: [BatchGenerationEngine.Seq], context: ModelContext, cancelled: Set<Int>
) {
    guard let model = context.model as? BatchableModel else {
        for s in batch {
            s.continuation.yield(.info(infoFor(s)))
            s.continuation.finish()
        }
        return
    }
    for s in batch where s.detokenizer == nil {
        s.detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
    }

    let lengths = batch.map { $0.promptTokens.count }
    let maxLen = lengths.max() ?? 1
    let leftPad = lengths.map { maxLen - $0 }
    let padded = batch.map {
        Array(repeating: Int32(0), count: maxLen - $0.promptTokens.count) + $0.promptTokens
    }
    let promptArray = MLXArray(padded.flatMap { $0 }, [batch.count, maxLen])
    let cache = model.newBatchCache(leftPadding: leftPad)

    // Prefill in CHUNKS, not one forward. A single forward over the full padded prompt materializes
    // an O(maxLen²) attention score tensor — at ~32k tokens that's a ~49GB buffer, which exceeds
    // Metal's max buffer size and HARD-CRASHES the process (metal::malloc fatal error). Chunking
    // bounds peak memory to O(chunk·maxLen) and matches the MTP path's chunked prefill.
    let prefillChunk = 512
    var lastLogits = MLXArray.zeros([batch.count, 1])
    var pos = 0
    while pos < maxLen {
        let n = min(prefillChunk, maxLen - pos)
        let chunk = promptArray[0..., pos ..< (pos + n)]
        let logits = model.batchForward(chunk, cache: cache)
        if pos + n >= maxLen { lastLogits = logits[0..., -1, 0...] }  // final chunk's last position
        eval(cache.map { $0.state }.flatMap { $0 })  // realize per chunk; bound peak memory + graph
        pos += n
    }
    let maxTokens = batch.map(\.maxTokens).max() ?? 0

    var live = batch
    var step = 0
    while !live.isEmpty && step < maxTokens {
        let tokens = sampleBatch(lastLogits, sequences: live)
        eval(tokens)
        let toks = tokens.asArray(Int32.self)

        var keepIndices: [Int32] = []
        var nextLive: [BatchGenerationEngine.Seq] = []
        var nextInputs: [Int32] = []
        for (i, s) in live.enumerated() {
            if cancelled.contains(s.id) { s.continuation.finish(); continue }
            let tok = toks[i]
            if s.stopTokenIds.contains(Int(tok)) || s.generated >= s.maxTokens {
                emitTail(s)
                s.continuation.yield(.info(infoFor(s)))
                s.continuation.finish()
                continue
            }
            s.detokenizer?.append(token: Int(tok))
            if let chunk = s.detokenizer?.next() { s.continuation.yield(.chunk(chunk)) }
            s.generated += 1
            s.lastToken = tok
            keepIndices.append(Int32(i))
            nextLive.append(s)
            nextInputs.append(tok)
        }

        if nextLive.count != live.count, !keepIndices.isEmpty {
            let idx = MLXArray(keepIndices)
            for c in cache {
                (c as? BatchKVCache)?.filter(batchIndices: idx)
                (c as? MambaCache)?.filter(batchIndices: idx)
            }
        }
        live = nextLive
        if live.isEmpty { break }

        let inputArray = MLXArray(nextInputs, [nextInputs.count, 1])
        lastLogits = model.batchForward(inputArray, cache: cache)[0..., -1, 0...]
        step += 1
    }

    for s in live where !cancelled.contains(s.id) {
        emitTail(s)
        s.continuation.yield(.info(infoFor(s, reason: .length)))
        s.continuation.finish()
    }
}

private func emitTail(_ s: BatchGenerationEngine.Seq) {
    if let tail = s.detokenizer?.next(), !tail.isEmpty {
        s.continuation.yield(.chunk(tail))
    }
}

private func infoFor(_ s: BatchGenerationEngine.Seq, reason: GenerateStopReason = .stop)
    -> GenerateCompletionInfo
{
    GenerateCompletionInfo(
        promptTokenCount: s.promptTokens.count, generationTokenCount: s.generated,
        promptTime: 0, generationTime: 0, stopReason: reason)
}

/// Per-row sampling: greedy when temperature==0 (whole-batch argMax fast path), else per-row temp.
private func sampleBatch(_ logits: MLXArray, sequences: [BatchGenerationEngine.Seq]) -> MLXArray {
    if sequences.allSatisfy({ $0.temperature == 0 }) {
        return argMax(logits, axis: -1).asType(.int32)
    }
    var perRow: [MLXArray] = []
    for (i, s) in sequences.enumerated() {
        let row = logits[i ..< (i + 1), 0...]
        perRow.append(
            s.temperature == 0
                ? argMax(row, axis: -1).asType(.int32)
                : categorical(row * (1 / s.temperature)).asType(.int32))
    }
    return concatenated(perRow, axis: 0).reshaped([sequences.count])
}
