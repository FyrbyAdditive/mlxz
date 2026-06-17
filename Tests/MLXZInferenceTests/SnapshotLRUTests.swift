import Testing

@testable import MLXZInference

/// The LRU must keep multiple prefix snapshots so an unrelated request can't evict a valuable one —
/// the exact regression: VS Code's title-generation between chat turns clobbered the single shared
/// system-prompt snapshot. Logic here depends only on the token prefixes (caches are empty stand-ins).
@Suite struct SnapshotLRUTests {
    /// A long shared "system prompt" prefix (≥ minReuse=16) plus a distinguishing tail.
    private func sysPrefix(_ tail: [Int32] = []) -> [Int32] { Array(0 ..< 40).map(Int32.init) + tail }

    @Test func bestMatchReturnsLongestExactPrefix() {
        let lru = SnapshotLRU(capacity: 4)
        lru.insert(tokens: sysPrefix(), modelCache: [], mtpCache: [])
        // A new prompt that extends the system prefix → reuse the whole 40-token snapshot.
        let newTokens = sysPrefix([99, 98, 97])
        let m = lru.bestMatch(for: newTokens)
        #expect(m != nil)
        #expect(m?.tokens.count == 40)
    }

    @Test func noMatchForUnrelatedPrompt() {
        let lru = SnapshotLRU(capacity: 4)
        lru.insert(tokens: sysPrefix(), modelCache: [], mtpCache: [])
        // A short unrelated prompt shares no ≥16 prefix.
        #expect(lru.bestMatch(for: [500, 501, 502]) == nil)
    }

    @Test func unrelatedInsertDoesNotEvictValuableSnapshot() {
        // THE regression test: cache the system prompt, then a title-gen prompt, then a follow-up
        // that extends the system prompt — it must still find the system-prompt snapshot.
        let lru = SnapshotLRU(capacity: 4)
        lru.insert(tokens: sysPrefix(), modelCache: [], mtpCache: [])
        lru.insert(tokens: [700, 701, 702, 703], modelCache: [], mtpCache: [])  // title-gen
        let follow = sysPrefix([1, 2])
        let m = lru.bestMatch(for: follow)
        #expect(m?.tokens.count == 40, "system-prompt snapshot was evicted by an unrelated prompt")
    }

    @Test func capacityEvictsLeastRecentlyUsed() {
        let lru = SnapshotLRU(capacity: 2)
        let a = Array(0 ..< 20).map(Int32.init)
        let b = Array(100 ..< 120).map(Int32.init)
        let c = Array(200 ..< 220).map(Int32.init)
        lru.insert(tokens: a, modelCache: [], mtpCache: [])
        lru.insert(tokens: b, modelCache: [], mtpCache: [])
        lru.insert(tokens: c, modelCache: [], mtpCache: [])  // evicts a (LRU)
        #expect(lru.bestMatch(for: a + [1]) == nil)          // a gone
        #expect(lru.bestMatch(for: b + [1]) != nil)
        #expect(lru.bestMatch(for: c + [1]) != nil)
    }

    @Test func bestMatchTouchesMRUSoItSurvivesEviction() {
        let lru = SnapshotLRU(capacity: 2)
        let a = Array(0 ..< 20).map(Int32.init)
        let b = Array(100 ..< 120).map(Int32.init)
        lru.insert(tokens: a, modelCache: [], mtpCache: [])
        lru.insert(tokens: b, modelCache: [], mtpCache: [])
        _ = lru.bestMatch(for: a + [1])  // touch a → a becomes MRU
        let c = Array(200 ..< 220).map(Int32.init)
        lru.insert(tokens: c, modelCache: [], mtpCache: [])  // should evict b (now LRU), not a
        #expect(lru.bestMatch(for: a + [1]) != nil, "touched entry should survive")
        #expect(lru.bestMatch(for: b + [1]) == nil, "untouched entry should be evicted")
    }

    @Test func capacityZeroDisablesCaching() {
        let lru = SnapshotLRU(capacity: 0)
        lru.insert(tokens: sysPrefix(), modelCache: [], mtpCache: [])
        #expect(lru.isEmpty)
        #expect(lru.bestMatch(for: sysPrefix([1])) == nil)
    }
}
