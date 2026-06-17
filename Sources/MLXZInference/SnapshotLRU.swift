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
    /// One cached prefix snapshot: the caches plus the exact token prefix they encode.
    struct Entry {
        let tokens: [Int32]
        let modelCache: [KVCache]
        let mtpCache: [KVCache]
    }

    /// Front = most-recently-used. Capacity 0 disables caching entirely.
    private var entries: [Entry] = []
    private let capacity: Int

    init(capacity: Int) {
        self.capacity = max(0, capacity)
    }

    var isEmpty: Bool { entries.isEmpty }

    /// Number of cached snapshots.
    var count: Int { entries.count }

    /// Drop all cached snapshots (e.g. on model unload).
    func clear() { entries.removeAll() }

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

    /// Insert a new snapshot at the front, replacing any existing entry that encodes the same prefix
    /// (so re-running an identical prefix refreshes rather than duplicates), then evict LRU over
    /// capacity. Additive: inserting a short unrelated prefix never removes a still-cached larger one
    /// unless capacity forces it.
    func insert(tokens: [Int32], modelCache: [KVCache], mtpCache: [KVCache]) {
        guard capacity > 0 else { return }
        entries.removeAll { $0.tokens == tokens }
        entries.insert(Entry(tokens: tokens, modelCache: modelCache, mtpCache: mtpCache), at: 0)
        if entries.count > capacity {
            entries.removeLast(entries.count - capacity)
        }
    }

    /// The token sequence of the MRU entry, used to pick the next snapshot boundary (the stable
    /// region a future request is most likely to share). Empty when the cache is empty.
    var mostRecentTokens: [Int32] { entries.first?.tokens ?? [] }
}
