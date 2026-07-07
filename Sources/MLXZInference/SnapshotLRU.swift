import Foundation
import MLXLMCommon

/// A small LRU of prefix-snapshot KV caches for cross-request reuse on the MTP path.
///
/// The previous design kept a SINGLE snapshot, so any unrelated request (e.g. VS Code's
/// fire-and-forget title-generation between chat turns) evicted a valuable one — the next real turn
/// then re-prefilled the whole system prompt (measured: 0.37s → 2.6s TTFT). This keeps up to
/// `capacity` snapshots keyed by the exact token prefix each encodes, evicting strictly
/// least-recently-used. A one-off short prompt at worst occupies one slot and ages out; it never
/// evicts a still-in-use larger snapshot before the others.
///
/// Non-Sendable contents (`KVCache` wraps `MLXArray`); `@unchecked Sendable` because every access
/// happens inside the owning `ModelContainer.perform`, and the gate runs the MTP path at
/// `maxConcurrency == 1`, so there is never concurrent access.
final class SnapshotLRU: @unchecked Sendable {
    /// One cached prefix snapshot: the caches plus the exact token prefix they encode, and its
    /// measured byte size (sum of cache state-array bytes) so the LRU can bound total memory.
    struct Entry {
        let tokens: [Int32]
        let modelCache: [KVCache]
        let mtpCache: [KVCache]
        let bytes: Int
    }

    /// Front = most-recently-used.
    private var entries: [Entry] = []
    /// Max number of snapshots (a coarse guard). 0 disables caching entirely.
    private let capacity: Int
    /// Hard ceiling on total bytes pinned by all snapshots. Eviction (LRU) keeps the sum ≤ this so
    /// RAM is bounded REGARDLESS of context length — a single 55k-token snapshot can be ~0.5–1.8GB,
    /// so without a byte cap a few of them blow memory to tens of GB. 0 disables the byte cap.
    private let maxBytes: Int

    init(capacity: Int, maxBytes: Int) {
        self.capacity = max(0, capacity)
        self.maxBytes = max(0, maxBytes)
    }

    var isEmpty: Bool { entries.isEmpty }

    /// Number of cached snapshots.
    var count: Int { entries.count }

    /// Total bytes currently pinned by all snapshots.
    var totalBytes: Int { entries.reduce(0) { $0 + $1.bytes } }

    /// Drop all cached snapshots (e.g. on model unload).
    func clear() { entries.removeAll() }

    /// Sum of the `state` array bytes across a snapshot's model + MTP caches.
    static func snapshotBytes(model: [KVCache], mtp: [KVCache]) -> Int {
        (model + mtp).reduce(0) { acc, kv in
            acc + kv.state.reduce(0) { $0 + $1.nbytes }
        }
    }

    /// The longest cached snapshot that is an exact prefix of `newTokens` (≥ `minReuse` tokens and
    /// strictly shorter than `newTokens`, matching `MTPCacheReuse.reuseCount`). Moves the chosen
    /// entry to the front (most-recently-used). Returns nil if nothing reusable.
    func bestMatch(for newTokens: [Int32], minReuse: Int = 16) -> Entry? {
        guard capacity > 0 else { return nil }
        var bestIndex: Int? = nil
        var bestLen = 0
        for (i, e) in entries.enumerated() {
            let n = MTPCacheReuse.reuseCount(
                snapshotTokens: e.tokens, newTokens: newTokens, minReuse: minReuse)
            if n > bestLen {
                bestLen = n
                bestIndex = i
            }
        }
        guard let idx = bestIndex else { return nil }
        let entry = entries.remove(at: idx)
        entries.insert(entry, at: 0)  // touch → MRU
        return entry
    }

    /// Insert a new snapshot at the front, replacing any existing entry with the same prefix, then
    /// evict least-recently-used entries until BOTH the count cap and the byte budget hold. The byte
    /// budget is the hard memory ceiling: a long-context snapshot can be ~GB, so eviction by bytes
    /// (not just count) keeps total RAM bounded regardless of prompt length.
    func insert(tokens: [Int32], modelCache: [KVCache], mtpCache: [KVCache]) {
        guard capacity > 0 else { return }
        let bytes = Self.snapshotBytes(model: modelCache, mtp: mtpCache)
        entries.removeAll { $0.tokens == tokens }
        entries.insert(
            Entry(tokens: tokens, modelCache: modelCache, mtpCache: mtpCache, bytes: bytes), at: 0)
        // Evict LRU (from the back) until under the count cap AND the byte budget. Always keep at
        // least the just-inserted MRU entry (index 0), even if it alone exceeds the budget.
        while entries.count > capacity { entries.removeLast() }
        if maxBytes > 0 {
            while entries.count > 1, totalBytes > maxBytes { entries.removeLast() }
        }
    }

    /// The token sequence of the MRU entry, used to pick the next snapshot boundary (the stable
    /// region a future request is most likely to share). Empty when the cache is empty.
    var mostRecentTokens: [Int32] { entries.first?.tokens ?? [] }

    /// The entry sharing the longest COMMON PREFIX with `newTokens` (the snapshot need not be an
    /// exact prefix — trim-sound caches can be copied and TRIMMED back to the shared point). For
    /// whole-generation snapshots this is essential: chat templates re-render prior turns (e.g.
    /// stripping reasoning), so the snapshot's tail never matches but the shared region is nearly
    /// the whole previous conversation. Returns the entry and the usable common length
    /// (< newTokens.count, ≥ minReuse). Touches the entry (MRU).
    func bestCommonPrefix(for newTokens: [Int32], minReuse: Int = 16) -> (entry: Entry, common: Int)? {
        guard capacity > 0 else { return nil }
        var bestIndex: Int? = nil
        var bestLen = 0
        for (i, e) in entries.enumerated() {
            let n = min(
                MTPCacheReuse.commonPrefixLength(e.tokens, newTokens), newTokens.count - 1)
            if n > bestLen {
                bestLen = n
                bestIndex = i
            }
        }
        guard let idx = bestIndex, bestLen >= minReuse else { return nil }
        let entry = entries.remove(at: idx)
        entries.insert(entry, at: 0)
        return (entry, bestLen)
    }
}
