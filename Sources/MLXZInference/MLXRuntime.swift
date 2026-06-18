import Foundation
import MLX
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

        if let mb = perf.gpuCacheLimitMB, mb >= 0 {
            MLX.GPU.set(cacheLimit: mb * 1024 * 1024)
        }
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
