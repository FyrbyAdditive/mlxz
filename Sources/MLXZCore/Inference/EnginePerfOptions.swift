import Foundation

/// Operational performance/memory options applied to every generation by an engine.
/// These are server-level tuning (not per-request) — they trade a little quality for
/// substantially less KV-cache memory, enabling longer contexts and larger MoE models.
public struct EnginePerfOptions: Sendable, Equatable {
    /// Quantize the full-attention KV cache to this many bits. Default 4 (verified: greedy output
    /// byte-identical to fp16 on the 27B, decode speed unchanged). `nil` = full precision. Only the
    /// attention layers are quantized; the GatedDeltaNet recurrent (Mamba) state stays full precision.
    /// Note: the absolute memory saving is small at typical context lengths (the hybrid model has few
    /// attention layers, so the attention KV is a minor fraction) — measured negligible for the
    /// cross-request prefix snapshots. It's lossless and free here, and provides headroom at very long
    /// contexts / many cached slots. Small models may degrade at low bit-widths — raise to 8 or nil.
    public var kvBits: Int?
    /// Group size for KV quantization.
    public var kvGroupSize: Int
    /// Token offset at which to start quantizing the KV cache (keep the early context exact).
    public var quantizedKVStart: Int
    /// Cap the KV cache to this many tokens (uses a rotating cache). nil = unbounded.
    public var maxKVSize: Int?
    /// Reuse the KV cache for a shared prompt prefix across requests.
    public var prefixCache: Bool

    /// Use native MTP self-speculative decoding when the loaded model has an MTP head.
    /// Pure speedup (identical output); on by default for MTP-capable models.
    public var useMTP: Bool

    /// Upper bound on MLX's GPU buffer cache, in MB. MLX defaults its cache to the (large) memory
    /// limit, so it can hoard many GB of buffers alongside a multi-GB model — driving memory
    /// pressure and weight eviction (a silent, catastrophic slowdown). MLX's own docs recommend a
    /// much lower cap for memory-constrained / long-inference workloads (small caches often perform
    /// identically). nil = leave MLX's default. Default: 512 MB.
    public var gpuCacheLimitMB: Int?

    /// Max sequences decoded together in one batched forward pass (continuous batching) for plain
    /// (non-MTP) requests. >1 lets concurrent requests run concurrently instead of being
    /// serialized/rejected. 1 effectively serializes. Default 8.
    public var maxBatch: Int

    /// Hard ceiling, in MB, on total memory pinned by the prefix-snapshot LRU. A long-context
    /// snapshot can be hundreds of MB–GB, so the LRU evicts least-recently-used snapshots to keep the
    /// sum under this — bounding RAM REGARDLESS of context length or number of conversations. Default
    /// 2048 (2GB). 0 = no byte cap (count cap only — not recommended for long contexts).
    public var prefixCacheBytesMB: Int

    /// Token granularity at which prefix snapshots are captured during prefill (block-aligned). A
    /// future request sharing a prefix reuses the largest block boundary ≤ the shared length, so
    /// smaller = more reuse coverage but more snapshots (more LRU RAM); larger = coarser reuse, less
    /// RAM. Default 512. Only used when `prefixCache` is true.
    public var snapshotBlock: Int

    /// Number of prefix-snapshot slots in the MTP cross-request cache (LRU). Multi-slot so an
    /// unrelated request (e.g. an IDE's title-generation between chat turns) can't evict a valuable
    /// system-prompt snapshot, and so each request's block-boundary snapshots coexist with other
    /// conversations'. 1 = single-slot; 0 = disable cross-request reuse. Each snapshot is small for
    /// the hybrid model (measured ~4.5MB at 10k tokens — the GatedDeltaNet state is O(1) and the
    /// attention KV is 4-bit-quantized), so a generous count is cheap. Default 16. Used when
    /// `prefixCache` is true.
    public var prefixCacheSlots: Int

    public init(
        kvBits: Int? = 4,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        maxKVSize: Int? = nil,
        prefixCache: Bool = true,
        useMTP: Bool = true,
        gpuCacheLimitMB: Int? = 512,
        maxBatch: Int = 8,
        prefixCacheSlots: Int = 16,
        prefixCacheBytesMB: Int = 2048,
        snapshotBlock: Int = 512
    ) {
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.maxKVSize = maxKVSize
        self.prefixCache = prefixCache
        self.useMTP = useMTP
        self.gpuCacheLimitMB = gpuCacheLimitMB
        self.maxBatch = maxBatch
        self.prefixCacheSlots = prefixCacheSlots
        self.prefixCacheBytesMB = prefixCacheBytesMB
        self.snapshotBlock = snapshotBlock
    }

    public static let `default` = EnginePerfOptions()
}
