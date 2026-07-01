import Foundation
import ArgumentParser
import Logging
import MLXZCore
import MLXZInference
import MLXZServer
import MLXZHub

/// Headless OpenAI-compatible server. The composition root: wires the real MLX loader into
/// the ModelManager and starts the Hummingbird server. Used for curl/SDK smoke tests and as
/// the engine behind the GUI.
@main
struct MLXZServe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlxz-serve",
        abstract: "Serve a local MLX model over an OpenAI-compatible API."
    )

    @Option(name: .long, help: "HuggingFace repo id, e.g. mlx-community/Qwen3.6-4B-4bit")
    var model: String

    @Option(name: .long, help: "Bind host (use 0.0.0.0 for LAN).")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Bind port.")
    var port: Int = 8080

    @Option(name: .long, help: "Optional API key required in the Authorization: Bearer header.")
    var apiKey: String?

    @Option(name: .long, help: "Quantize the attention KV cache (and prefix snapshots) to N bits. Default 4 (verified lossless for greedy on the 27B, ~4x smaller KV). Use 8 for small models; 0 disables (full precision).")
    var kvBits: Int = 4

    @Option(name: .long, help: "Cap the KV cache to N tokens (rotating cache); bounds memory on long chats. Disables prefix-cache reuse.")
    var maxKvSize: Int?

    @Flag(name: .long, inversion: .prefixedNo, help: "Reuse the KV cache for shared prompt prefixes across requests.")
    var prefixCache: Bool = true

    @Flag(name: .long, inversion: .prefixedNo, help: "Use native MTP self-speculative decoding for MTP-capable models.")
    var mtp: Bool = true

    @Option(name: .long, help: "Attach a standalone MTP drafter checkpoint (repo id) to the model for draft-model speculative decoding.")
    var mtpDraft: String?

    @Option(name: .long, help: "Bound MLX's GPU buffer cache to N MB (default 512). Prevents MLX hoarding multi-GB of scratch buffers next to a large model. 0 disables the cache.")
    var gpuCacheMb: Int = 512

    @Option(name: .long, help: "Apply a process-wide wired-memory limit of N MB after load so the resident model weights aren't paged/evicted under memory pressure (clamped to the device's recommended working set). 0 = off (default). The credible lever for bandwidth-bound decode on a memory-constrained machine.")
    var wiredMb: Int = 0

    @Option(name: .long, help: "Max concurrent requests decoded together in one batched forward pass (continuous batching) for GatedDeltaNet models. Default 8. Concurrent requests run together instead of being serialized/rejected. MTP models decode single-sequence and queue.")
    var maxBatch: Int = 8

    @Option(name: .long, help: "Max requests waiting for a generation slot before returning 429. Default 0 = unbounded (always queue, never reject).")
    var maxQueue: Int = 0

    @Option(name: .long, help: "Prompt tokens per prefill forward pass (chunked prefill). Larger = fewer GPU syncs and better tensor-core utilization (faster TTFT on long prompts); smaller = lower peak memory. 0 = fork default (512).")
    var prefillChunk: Int = 0

    @Option(name: .long, help: "Prefix-snapshot cache slots (LRU) for cross-request reuse on the MTP path. Holds block-boundary snapshots so conversations sharing a system prompt reuse it. Each snapshot is small (~4.5MB at 10k tok, 4-bit KV). Default 16; 0 disables reuse.")
    var prefixCacheSlots: Int = 16

    @Option(name: .long, help: "Token granularity for prefix-snapshot capture (block-aligned). Smaller = the single snapshot lands closer to the shared-prefix boundary (more reuse); larger = coarser. Default 512.")
    var snapshotBlock: Int = 512

    @Option(name: .long, help: "Hard ceiling (MB) on total RAM pinned by the prefix-snapshot LRU. Evicts least-recently-used snapshots to stay under this, bounding memory regardless of context length. Default 2048; 0 = no byte cap.")
    var prefixCacheMb: Int = 2048

    @Option(name: .long, help: "Default cap on <think> reasoning tokens before the block is force-closed and the model must answer. Bounds the worst case (reasoning can otherwise run thousands of tokens). Default 2048; 0 = uncapped. A request's reasoning_effort/max_reasoning_tokens overrides this.")
    var reasoningBudget: Int = 2048

    @Flag(name: .long, help: "Print the VS Code Copilot model-config snippet and exit.")
    var printCopilotConfig: Bool = false

    @Flag(name: .long, help: "Download --model via the exact GUI download path, printing live progress ticks, then exit. Verifies progress is reported through large (Xet) files.")
    var downloadTest: Bool = false

    // MARK: - Benchmark mode (Phase 0 harness)

    @Flag(name: .long, help: "Run a fixed-prompt benchmark (no server) and print median decode/TTFT/memory metrics, then exit. Reuses the exact server load + generate path so perf knobs are exercised identically.")
    var bench: Bool = false

    @Option(name: .long, help: "Benchmark prompt size in tokens (a repeated filler prompt). Small (~512) isolates decode; large (~80000) isolates prefill/TTFT. Default 512.")
    var promptTokens: Int = 512

    @Option(name: .long, help: "Benchmark tokens to generate per iteration. Default 256.")
    var benchMaxTokens: Int = 256

    @Option(name: .long, help: "Benchmark warm iterations (after 1 discarded warmup). Default 3.")
    var iters: Int = 3

    func run() async throws {
        if downloadTest {
            // Exercise the exact GUI download path (MLXModelDownloader) and print progress ticks so we
            // can see continuous progress through large (Xet-transported) safetensors files.
            let dl = MLXModelDownloader()
            var lastPct = -1
            try await dl.download(ModelDescriptor(repoID: model)) { @MainActor p in
                let pct = Int(p.fraction * 100)
                if pct != lastPct {
                    lastPct = pct
                    let mb = Double(p.completedBytes) / 1_000_000
                    let tot = Double(p.totalBytes) / 1_000_000
                    FileHandle.standardError.write(Data(
                        String(format: "[DL] %3d%%  %.1f / %.1f MB\n", pct, mb, tot).utf8))
                }
            }
            FileHandle.standardError.write(Data("[DL] done\n".utf8))
            return
        }
        // Force the per-step decode diagnostic on for the benchmark before any model/MTPSession is
        // touched (MTPSession reads MLXZ_DECODE_DIAG once at static init).
        if bench { setenv("MLXZ_DECODE_DIAG", "1", 1) }
        try await runImpl()
    }

    func runImpl() async throws {
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardOutput(label: label)
            handler.logLevel = .info
            return handler
        }
        let logger = Logger(label: "mlxz")

        let descriptor = ModelDescriptor(repoID: model)

        if printCopilotConfig {
            let caps = ModelCapabilityDetector.detect(repoID: model)
            print(CopilotConfig.setupHint(host: host, port: port))
            print()
            print(CopilotConfig.modelEntry(repoID: model, host: host, port: port, capabilities: caps))
            return
        }

        let perf = EnginePerfOptions(
            kvBits: kvBits > 0 ? kvBits : nil,  // 0 = full precision
            maxKVSize: maxKvSize,
            prefixCache: prefixCache,
            useMTP: mtp,
            gpuCacheLimitMB: gpuCacheMb,
            wiredLimitMB: wiredMb > 0 ? wiredMb : nil,
            maxBatch: maxBatch,
            prefillChunk: prefillChunk,
            prefixCacheSlots: prefixCacheSlots,
            prefixCacheBytesMB: prefixCacheMb,
            snapshotBlock: snapshotBlock,
            reasoningTokenBudget: reasoningBudget > 0 ? reasoningBudget : nil
        )
        let manager = ModelManager(
            loader: MLXModelLoader(perf: perf, draftModelID: mtpDraft), logger: logger)

        logger.info("loading model (first run downloads from HuggingFace)…", metadata: ["model": .string(model)])
        try await manager.load(descriptor)
        logger.info("model ready", metadata: ["model": .string(model)])

        // Phase 1A: apply the wired-memory limit AFTER load (weights already resident) so they
        // aren't paged under pressure. No-op when --wired-mb is 0.
        if let applied = await MLXRuntime.configureWired(perf: perf) {
            logger.info("wired-memory limit applied", metadata: ["bytes": .stringConvertible(applied)])
        }

        if bench {
            try await runBenchmark(manager: manager, perf: perf)
            return
        }

        let localStore = LocalModelStore()
        let embeddingManager = EmbeddingManager(loader: MLXEmbeddingLoader())
        let server = InferenceServer(
            manager: manager,
            maxConcurrent: maxBatch,
            maxWaiting: maxQueue,
            logger: logger,
            logSink: { line in logger.info("\(line)") },
            extraModelIDs: { localStore.installedModels().map(\.descriptor.repoID) },
            embeddingManager: embeddingManager,
            metricsSink: { usage in
                if let tps = usage.tokensPerSecond {
                    logger.info("generated \(usage.completionTokens) tokens at \(String(format: "%.1f", tps)) tok/s")
                }
            }
        )
        try await server.start(ServerConfig(host: host, port: port, apiKey: apiKey))

        // Print a ready banner with the Copilot snippet.
        let caps = await manager.currentEngine()?.capabilities ?? [.chat, .tools]
        print("\n\(CopilotConfig.setupHint(host: host, port: port))\n")
        print(CopilotConfig.modelEntry(repoID: model, host: host, port: port, capabilities: caps))
        print("\nServing. Press Ctrl-C to stop.\n")

        // Run until cancelled (Ctrl-C).
        try await Task.sleep(for: .seconds(60 * 60 * 24 * 365))
    }

    // MARK: - Benchmark routine

    /// One iteration's measured numbers. Decode ms/step comes from the engine's `[DECODE]` diagnostic
    /// (printed to stderr); here we capture the end-to-end wall split (prefill = time-to-first-token,
    /// decode = remaining) plus token count and GPU memory snapshot.
    private struct BenchRun {
        var ttftSeconds: Double          // wall time to first generated token (prefill-dominated)
        var decodeSeconds: Double        // wall time generating the rest
        var generatedTokens: Int
        var peakMemoryBytes: Int
        var decodeTokPerSec: Double { decodeSeconds > 0 ? Double(max(0, generatedTokens - 1)) / decodeSeconds : 0 }
    }

    /// Write a benchmark line to stderr (unbuffered, so it interleaves correctly with the engine's
    /// `[DECODE]` stderr lines and always flushes — plain `print` to stdout buffers under redirection).
    private func benchPrint(_ s: String) {
        FileHandle.standardError.write(Data((s + "\n").utf8))
    }

    private func runBenchmark(manager: ModelManager, perf: EnginePerfOptions) async throws {
        guard let engine = await manager.currentEngine() else {
            benchPrint("bench: no engine loaded"); return
        }
        // A deterministic filler prompt of approximately `promptTokens` tokens (~0.75 word/token).
        let words = max(1, Int(Double(promptTokens) * 0.75))
        let filler = Array(repeating: "lorem ipsum dolor sit amet", count: (words / 5) + 1)
            .joined(separator: " ")
        let prompt = "Repeat nothing. Context follows.\n\(filler)\n\nReply with a short story."
        let request = GenerationRequest(
            messages: [ChatMessage(role: .user, text: prompt)],
            sampling: SamplingParameters(temperature: 0),   // greedy → deterministic, comparable
            maxTokens: benchMaxTokens,
            reasoningTokenBudget: 0                          // uncapped: measure raw decode, not the cap
        )

        benchPrint("=== mlxz bench ===")
        benchPrint("model=\(model) mtpDraft=\(mtpDraft ?? "none") promptTokens≈\(promptTokens) maxTokens=\(benchMaxTokens) iters=\(iters) (+1 warmup)")
        benchPrint("perf: kvBits=\(kvBits) gpuCacheMB=\(gpuCacheMb) wiredMB=\(wiredMb > 0 ? String(wiredMb) : "off") prefixCacheSlots=\(prefixCacheSlots)")

        var runs: [BenchRun] = []
        for i in 0 ..< (iters + 1) {
            MLXRuntime.resetPeakMemory()
            let run = try await benchOnce(engine: engine, request: request)
            let tag = i == 0 ? "warmup(discard)" : "run \(i)"
            benchPrint(String(format: "  %@: ttft=%.2fs decode=%.2fs gen=%d tok/s=%.2f peakMem=%.2fGB",
                              tag, run.ttftSeconds, run.decodeSeconds, run.generatedTokens,
                              run.decodeTokPerSec, Double(run.peakMemoryBytes) / 1e9))
            if i > 0 { runs.append(run) }   // discard the first (warmup)
        }
        guard !runs.isEmpty else { return }

        func median(_ xs: [Double]) -> Double {
            let s = xs.sorted(); let n = s.count
            return n % 2 == 1 ? s[n/2] : (s[n/2 - 1] + s[n/2]) / 2
        }
        func stddev(_ xs: [Double]) -> Double {
            guard xs.count > 1 else { return 0 }
            let m = xs.reduce(0, +) / Double(xs.count)
            return (xs.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(xs.count - 1)).squareRoot()
        }

        let tps = runs.map(\.decodeTokPerSec)
        let ttft = runs.map(\.ttftSeconds)
        let dec = runs.map(\.decodeSeconds)
        let peak = runs.map { Double($0.peakMemoryBytes) / 1e9 }
        let tpsMed = median(tps)
        let tpsVarPct = tpsMed > 0 ? stddev(tps) / tpsMed * 100 : 0

        benchPrint("--- median over \(runs.count) warm runs ---")
        benchPrint(String(format: "  decode tok/s = %.2f  (stddev/median = %.1f%%)", tpsMed, tpsVarPct))
        benchPrint(String(format: "  TTFT         = %.2fs", median(ttft)))
        benchPrint(String(format: "  decode time  = %.2fs", median(dec)))
        benchPrint(String(format: "  peak memory  = %.2f GB", median(peak)))
        benchPrint("  (per-step backbone/mtp/STEPWALL: see the [DECODE] lines above on stderr)")
        if tpsVarPct >= 3 {
            benchPrint("  ⚠️ run-to-run variance ≥3% — signal too noisy to trust a 5% win; rerun on a quiet machine.")
        }
    }

    /// Run one generation, splitting wall time into TTFT (to first token) and decode (the rest).
    private func benchOnce(engine: any InferenceEngine, request: GenerationRequest) async throws -> BenchRun {
        let t0 = Date()
        var firstTokenAt: Date?
        var generated = 0
        let stream = try await engine.generate(request)
        for try await event in stream {
            switch event {
            case .textDelta, .reasoningDelta:
                if firstTokenAt == nil { firstTokenAt = Date() }
                generated += 1
            case .completed(let result):
                generated = max(generated, result.usage.completionTokens)
            default:
                break
            }
        }
        let end = Date()
        let ttft = (firstTokenAt ?? end).timeIntervalSince(t0)
        let decode = end.timeIntervalSince(firstTokenAt ?? end)
        return BenchRun(
            ttftSeconds: ttft, decodeSeconds: decode, generatedTokens: generated,
            peakMemoryBytes: MLXRuntime.peakMemoryBytes)
    }
}
