import Foundation
import MLXZCore
import Observation

/// User-facing performance/memory settings, persisted in `UserDefaults`. Read into
/// `EnginePerfOptions` per model load (via the loader's perf provider), so changes apply on the
/// next model load. Defaults match `EnginePerfOptions`'s defaults.
@Observable
public final class PerfSettings: @unchecked Sendable {
    /// KV-cache quantization bits: 0 = full precision (fp16), else 4 or 8.
    public var kvBits: Int {
        didSet { defaults.set(kvBits, forKey: Keys.kvBits) }
    }
    /// Prefix-snapshot LRU slots for cross-request reuse (0 disables reuse).
    public var prefixCacheSlots: Int {
        didSet { defaults.set(prefixCacheSlots, forKey: Keys.prefixCacheSlots) }
    }
    /// Token granularity for block-aligned prefix-snapshot capture.
    public var snapshotBlock: Int {
        didSet { defaults.set(snapshotBlock, forKey: Keys.snapshotBlock) }
    }
    /// Hard RAM ceiling (MB) for the prefix-snapshot LRU.
    public var prefixCacheBytesMB: Int {
        didSet { defaults.set(prefixCacheBytesMB, forKey: Keys.prefixCacheBytesMB) }
    }
    /// Default cap on `<think>` reasoning tokens before the block is force-closed (0 = uncapped).
    /// A request's `reasoning_effort`/`max_reasoning_tokens` overrides this per request.
    public var reasoningTokenBudget: Int {
        didSet { defaults.set(reasoningTokenBudget, forKey: Keys.reasoningTokenBudget) }
    }
    /// Whether to auto-attach a matching installed MTP drafter (self-speculative decoding) when a
    /// model is loaded. Default true. Disable to load the base model alone (e.g. to A/B the speedup,
    /// or save the drafter's memory). Applies on the next model load.
    public var useMTPDrafter: Bool {
        didSet { defaults.set(useMTPDrafter, forKey: Keys.useMTPDrafter) }
    }
    /// Max image resolution (pixels = w×h) fed to the vision encoder; larger images are downscaled.
    /// Bounds the vision-token tensors that otherwise OOM the GPU on a high-res photo. Default ≈4 MP.
    public var maxImagePixels: Int {
        didSet { defaults.set(maxImagePixels, forKey: Keys.maxImagePixels) }
    }

    private let defaults: UserDefaults

    private enum Keys {
        static let kvBits = "perf.kvBits"
        static let prefixCacheSlots = "perf.prefixCacheSlots"
        static let snapshotBlock = "perf.snapshotBlock"
        static let prefixCacheBytesMB = "perf.prefixCacheBytesMB"
        static let reasoningTokenBudget = "perf.reasoningTokenBudget"
        static let useMTPDrafter = "perf.useMTPDrafter"
        static let maxImagePixels = "perf.maxImagePixels"
    }

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        let d = EnginePerfOptions.default
        // `object(forKey:)` distinguishes "unset" (use default) from an explicit 0.
        self.kvBits = (defaults.object(forKey: Keys.kvBits) as? Int) ?? (d.kvBits ?? 0)
        self.prefixCacheSlots =
            (defaults.object(forKey: Keys.prefixCacheSlots) as? Int) ?? d.prefixCacheSlots
        self.snapshotBlock =
            (defaults.object(forKey: Keys.snapshotBlock) as? Int) ?? d.snapshotBlock
        self.prefixCacheBytesMB =
            (defaults.object(forKey: Keys.prefixCacheBytesMB) as? Int) ?? d.prefixCacheBytesMB
        self.reasoningTokenBudget =
            (defaults.object(forKey: Keys.reasoningTokenBudget) as? Int) ?? (d.reasoningTokenBudget ?? 0)
        self.useMTPDrafter =
            (defaults.object(forKey: Keys.useMTPDrafter) as? Bool) ?? d.useMTP
        self.maxImagePixels =
            (defaults.object(forKey: Keys.maxImagePixels) as? Int) ?? d.maxImagePixels
    }

    /// Build `EnginePerfOptions` from the current settings, preserving all other engine defaults.
    public func engineOptions() -> EnginePerfOptions {
        EnginePerfOptions(
            kvBits: kvBits > 0 ? kvBits : nil,
            prefixCacheSlots: prefixCacheSlots,
            prefixCacheBytesMB: prefixCacheBytesMB,
            snapshotBlock: snapshotBlock,
            reasoningTokenBudget: reasoningTokenBudget > 0 ? reasoningTokenBudget : nil,
            maxImagePixels: maxImagePixels)
    }
}
