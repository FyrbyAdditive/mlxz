import Foundation

/// Operational performance/memory options applied to every generation by an engine.
/// These are server-level tuning (not per-request) — they trade a little quality for
/// substantially less KV-cache memory, enabling longer contexts and larger MoE models.
public struct EnginePerfOptions: Sendable, Equatable {
    /// Quantize the KV cache to this many bits (e.g. 8). nil = full precision (default).
    /// Large memory savings, but noticeably degrades small models — use only on large/MoE
    /// models that are memory-bound. Leave nil for best quality.
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

    public init(
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        maxKVSize: Int? = nil,
        prefixCache: Bool = true,
        useMTP: Bool = true,
        gpuCacheLimitMB: Int? = 512,
        maxBatch: Int = 8
    ) {
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.maxKVSize = maxKVSize
        self.prefixCache = prefixCache
        self.useMTP = useMTP
        self.gpuCacheLimitMB = gpuCacheLimitMB
        self.maxBatch = maxBatch
    }

    public static let `default` = EnginePerfOptions()
}
