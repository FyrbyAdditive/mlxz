import Foundation
import MLX
import MLXLMCommon
import MLXZCore

/// One-time, process-wide MLX runtime tuning. Call `configure(perf:)` once at startup (before
/// loading a model) from each composition root (the headless server and the GUI app).
public enum MLXRuntime {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var configured = false

    /// Apply MLX GPU memory policy. The key lever: bound the GPU buffer **cache** so MLX doesn't
    /// hoard multi-GB of scratch buffers next to a multi-GB model (the default cache limit equals
    /// the memory limit, ~1.5× the recommended working set). An unbounded cache competes with model
    /// weights for the unified-memory working set and causes eviction/thrashing — a silent, large
    /// slowdown on big models like the 27B. MLX's docs recommend a much lower cap here, and note
    /// small caches often match unbounded ones for inference.
    public static func configure(perf: EnginePerfOptions = .default) {
        lock.lock()
        defer { lock.unlock() }
        guard !configured else { return }
        configured = true

        // Empirically swept on the 27B+MTP (BASELINE.md Phase 1B): 512 MB is optimal. 1024/2048 MB
        // REGRESS decode ~22% (the cache competes with resident weights → eviction/thrash); 0/256 MB
        // churn ~4%. Do not raise the default without re-measuring.
        if let mb = perf.gpuCacheLimitMB, mb >= 0 {
            MLX.GPU.set(cacheLimit: mb * 1024 * 1024)
        }
    }

    // MARK: - Wired memory (Phase 1A: keep the model weights resident, avoid paging stalls)

    /// A long-lived wired-memory ticket held for the whole process lifetime. Acquired in
    /// `configureWired` and intentionally never ended, so the wired limit persists. The sync
    /// `Memory.withWiredLimit` is a no-op in this MLX build; the real mechanism is the
    /// `WiredMemoryManager` ticket system (the same one the standard `generate()` uses), so we hold
    /// a fixed-policy ticket rather than a scoped block.
    nonisolated(unsafe) private static var wiredTicket: WiredMemoryTicket?

    /// Apply a process-wide wired-memory limit so the resident model weights aren't paged/evicted
    /// under memory pressure (the only credible lever for the bandwidth-bound decode). Call once at
    /// startup AFTER the model is loaded (so the weights are already resident). `wiredLimitMB` nil or
    /// ≤0 → no wired limit (today's behavior). Clamped to the device's recommended working set so an
    /// over-large limit can't starve the rest of the system.
    @discardableResult
    public static func configureWired(perf: EnginePerfOptions) async -> Int? {
        guard let mb = perf.wiredLimitMB, mb > 0 else { return nil }
        var bytes = mb * 1024 * 1024
        let maxWS = Int(MLX.GPU.deviceInfo().maxRecommendedWorkingSetSize)
        if maxWS > 0 { bytes = min(bytes, maxWS) }
        // Hold the ticket forever (never `.end()`): the limit persists for the process lifetime.
        let ticket = WiredMemoryTicket(size: bytes, policy: WiredFixedPolicy(limit: bytes))
        let applied = await ticket.start()
        wiredTicket = ticket
        return applied
    }

    // MARK: - Memory introspection (keeps the MLX import boundary inside this module)

    /// Peak GPU memory (bytes) since the last `resetPeakMemory()`. Used by the benchmark harness.
    public static var peakMemoryBytes: Int { MLX.Memory.peakMemory }
    /// Currently-active GPU memory (bytes).
    public static var activeMemoryBytes: Int { MLX.Memory.activeMemory }
    /// GPU buffer-cache memory (bytes).
    public static var cacheMemoryBytes: Int { MLX.Memory.cacheMemory }
    /// Reset the peak-memory high-water mark (call before a measured run). The `Memory.peakMemory`
    /// setter ignores its value and resets the high-water mark.
    public static func resetPeakMemory() { MLX.Memory.peakMemory = 0 }
}
