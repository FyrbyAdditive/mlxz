import Testing
@testable import MLXZCore

@Suite struct PrefixCachePlanTests {
    @Test func freshWhenNoCache() {
        #expect(PrefixCachePlan.plan(cachedTokens: [], newTokens: [1, 2, 3]) == .fresh)
    }

    @Test func freshWhenNewIsEmpty() {
        #expect(PrefixCachePlan.plan(cachedTokens: [1, 2], newTokens: []) == .fresh)
    }

    @Test func reusesSharedPrefix() {
        // cached: a long shared system prompt; new: same prefix + new user turn.
        let shared: [Int32] = Array(0..<20)
        let cached = shared + [100, 101]            // old user turn
        let new = shared + [200, 201, 202]          // new user turn
        let d = PrefixCachePlan.plan(cachedTokens: cached, newTokens: new)
        #expect(d.reuse)
        #expect(d.reuseCount == 20)
        #expect(d.trimCount == cached.count - 20)   // trim the old user turn
    }

    @Test func noReuseBelowThreshold() {
        // Only 3 tokens shared (< minReuse default 8).
        let cached: [Int32] = [1, 2, 3, 9, 9]
        let new: [Int32] = [1, 2, 3, 7, 7, 7]
        #expect(PrefixCachePlan.plan(cachedTokens: cached, newTokens: new) == .fresh)
    }

    @Test func noReuseWhenNewIsPrefixOfCached() {
        // newTokens fully contained in the prefix → no new token to feed → fresh.
        let cached: [Int32] = Array(0..<20)
        let new: [Int32] = Array(0..<20)
        #expect(PrefixCachePlan.plan(cachedTokens: cached, newTokens: new) == .fresh)
    }

    @Test func reuseLeavesAtLeastOneNewToken() {
        let shared: [Int32] = Array(0..<10)
        let cached = shared
        let new = shared + [42]
        let d = PrefixCachePlan.plan(cachedTokens: cached, newTokens: new)
        #expect(d.reuse)
        #expect(d.reuseCount == 10)
        #expect(d.trimCount == 0)        // nothing to trim; just append the one new token
    }

    @Test func commonPrefixLengthBasics() {
        #expect(PrefixCachePlan.commonPrefixLength([1, 2, 3], [1, 2, 9]) == 2)
        #expect(PrefixCachePlan.commonPrefixLength([1, 2, 3], [9]) == 0)
        #expect(PrefixCachePlan.commonPrefixLength([], [1]) == 0)
    }
}
