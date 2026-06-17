import Foundation
import MLXLMCommon

/// Holds a persistent KV cache and the token ids it currently encodes, enabling prefix reuse
/// across requests. Used ONLY by the single-sequence paths (MTP / cachedStream); the gate runs
/// those at `maxConcurrency == 1`, so requests serialize and one request can't clobber another's
/// cached prefix mid-flight (concurrent requests queue at the gate instead). Non-Sendable (KVCache
/// wraps MLXArray) — mutated only inside the owning `ModelContainer.perform`.
///
/// Reference type so the engine (a struct) can carry and mutate it across calls.
/// `@unchecked Sendable`: only ever read/written inside the owning `ModelContainer.perform`,
/// and the depth-1 gate keeps single-sequence requests from overlapping on it.
final class PromptCacheBox: @unchecked Sendable {
    /// The live KV cache, or nil before the first generation.
    var cache: [KVCache]?
    /// The token ids currently encoded by `cache` (prompt + previously generated tokens).
    var tokens: [Int32] = []

    /// For the MTP path: an LRU of prefix snapshots (backbone + MTP-head caches + the exact token
    /// prefix each encodes), enabling whole-prefix reuse across requests. Multi-slot so an unrelated
    /// request (e.g. VS Code's title-generation) can't evict a valuable system-prompt snapshot — it
    /// just occupies its own slot and ages out. (The hybrid model's SSM layers preclude mid-sequence
    /// trimming, so only whole-prefix reuse is sound.) Built with the configured capacity.
    let snapshotLRU: SnapshotLRU

    /// The full prompt token sequence of the most recent MTP request, used to pick the next
    /// snapshot boundary (the prefix this turn shares with the previous turn — the stable system
    /// prompt). Kept separate from the snapshot LRU so the FIRST overlapping pair can still be
    /// detected before any snapshot exists (the snapshot is captured one turn behind by design).
    var lastPromptTokens: [Int32] = []

    init(prefixCacheSlots: Int = 4) {
        self.snapshotLRU = SnapshotLRU(capacity: prefixCacheSlots)
    }

    /// Reset to empty (e.g. after model unload or when reuse is impossible).
    func reset() {
        cache = nil
        tokens = []
        snapshotLRU.clear()
    }
}

enum PrefixMatch {
    /// Length of the longest common prefix between two token sequences.
    static func commonPrefixLength(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }
}
