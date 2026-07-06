import Foundation
import MLX
import MLXZCore
import MLXLMCommon

/// Microbenchmark for the speculative-decoding VERIFY step: the cost of one target-model forward
/// over M tokens (M = 1..maxM) against a warm KV cache of a given context length. Speculative
/// decoding (DSpark) only pays off if verifying M drafted tokens in one pass is cheaper than M
/// single-token decode steps; on Apple Silicon the verify cost grows with M (the qmv→qmm kernel
/// switch and per-token attention are not free), so the fitted line `verify(M) ≈ a + b·M` is the
/// input to the go/no-go ceiling projection and to choosing the draft-block cap.
///
/// Discipline (BASELINE.md): every timed region force-evaluates its outputs (`eval`) — MLX is lazy
/// and an unconsumed forward measures nothing. The cache is trimmed back after each pass so every
/// iteration sees the identical context length.
public enum VerifyCurveBench {
    public struct Point: Sendable {
        public let ctx: Int
        public let m: Int
        public let medianMs: Double
        public let iters: Int
    }

    /// Least-squares fit of ms = a + b·m for one context length.
    public struct Fit: Sendable {
        public let ctx: Int
        public let aMs: Double   // fixed per-step cost
        public let bMs: Double   // marginal cost per verified token
    }

    public static func fit(_ points: [Point]) -> [Fit] {
        Dictionary(grouping: points, by: \.ctx).map { ctx, pts in
            let n = Double(pts.count)
            let sx = pts.reduce(0.0) { $0 + Double($1.m) }
            let sy = pts.reduce(0.0) { $0 + $1.medianMs }
            let sxx = pts.reduce(0.0) { $0 + Double($1.m) * Double($1.m) }
            let sxy = pts.reduce(0.0) { $0 + Double($1.m) * $1.medianMs }
            let denom = n * sxx - sx * sx
            let b = denom != 0 ? (n * sxy - sx * sy) / denom : 0
            let a = (sy - b * sx) / n
            return Fit(ctx: ctx, aMs: a, bMs: b)
        }.sorted { $0.ctx < $1.ctx }
    }

    /// Run the curve on a loaded model. Prefills each context once (chunked, KV quantized exactly
    /// as production via `maybeQuantizeKVCache`), then times M-token forwards with trim-rollback.
    public static func run(
        container: ModelContainer,
        perf: EnginePerfOptions,
        repoID: String,
        contexts: [Int],
        maxM: Int,
        itersPerPoint: Int
    ) async throws -> [Point] {
        let params = MLXInferenceEngine.mapParameters(
            SamplingParameters(temperature: 0), maxTokens: 1, perf: perf, repoID: repoID)
        return try await container.perform { context in
            // Filler tokens, tiled to the largest context (content is irrelevant to cost).
            let filler = Array(repeating: "lorem ipsum dolor sit amet consectetur", count: 128)
                .joined(separator: " ")
            var tokens = context.tokenizer.encode(text: filler).map { Int32($0) }
            let need = (contexts.max() ?? 512) + maxM
            while tokens.count < need { tokens += tokens }
            tokens = Array(tokens.prefix(need))

            var points: [Point] = []
            for ctx in contexts.sorted() {
                var cache = context.model.newCache(parameters: params)
                guard canTrimPromptCache(cache) else {
                    FileHandle.standardError.write(Data(
                        "[VERIFY-CURVE] ctx=\(ctx): cache not trimmable (rotating?) — skipped\n".utf8))
                    continue
                }
                // Chunked prefill to `ctx`, quantizing KV in place as the production path does.
                let chunkSize = params.prefillStepSize
                var pos = 0
                while pos < ctx {
                    let take = min(chunkSize, ctx - pos)
                    let chunk = MLXArray(tokens[pos ..< (pos + take)]).expandedDimensions(axis: 0)
                    _ = context.model(chunk, cache: cache)
                    maybeQuantizeKVCache(
                        cache: &cache, kvBits: params.kvBits, kvGroupSize: params.kvGroupSize,
                        quantizedKVStart: params.quantizedKVStart)
                    eval(cache.map { $0.state }.flatMap { $0 })
                    pos += take
                }

                for m in 1 ... maxM {
                    let block = MLXArray(tokens[ctx ..< (ctx + m)]).expandedDimensions(axis: 0)
                    var samplesMs: [Double] = []
                    // 2 warmups (kernel compile/cache) + timed iterations.
                    for iter in 0 ..< (itersPerPoint + 2) {
                        let t0 = DispatchTime.now()
                        let logits = context.model(block, cache: cache)
                        // A real verify consumes per-position decisions: force argmax, not just logits.
                        let decisions = argMax(logits, axis: -1)
                        eval(decisions)
                        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
                        if iter >= 2 { samplesMs.append(ms) }
                        // Roll the appended M tokens back so every iteration sees the same context —
                        // the same trim the DSpark session performs on rejected draft suffixes.
                        trimPromptCache(cache, numTokens: m)
                    }
                    let sorted = samplesMs.sorted()
                    let median = sorted.count % 2 == 1
                        ? sorted[sorted.count / 2]
                        : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
                    points.append(Point(ctx: ctx, m: m, medianMs: median, iters: itersPerPoint))
                    FileHandle.standardError.write(Data(String(
                        format: "[VERIFY-CURVE] ctx=%d M=%d median=%.2fms (n=%d)\n",
                        ctx, m, median, itersPerPoint).utf8))
                }
            }
            return points
        }
    }
}
