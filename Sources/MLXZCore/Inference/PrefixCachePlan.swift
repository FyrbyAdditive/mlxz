import Foundation

/// Decides how to reuse a persistent KV cache given the tokens it already encodes and the new
/// request's full token sequence. Pure and dependency-free so the reuse policy is unit-testable
/// without MLX.
public enum PrefixCachePlan {
    public struct Decision: Equatable, Sendable {
        /// Number of leading tokens that can be reused from the existing cache.
        public var reuseCount: Int
        /// Number of tokens to trim off the end of the existing cache to reach `reuseCount`.
        public var trimCount: Int
        /// Whether the existing cache is usable at all (false → start fresh).
        public var reuse: Bool

        public init(reuseCount: Int, trimCount: Int, reuse: Bool) {
            self.reuseCount = reuseCount
            self.trimCount = trimCount
            self.reuse = reuse
        }

        public static let fresh = Decision(reuseCount: 0, trimCount: 0, reuse: false)
    }

    /// - Parameters:
    ///   - cachedTokens: tokens currently encoded by the cache (cache offset == cachedTokens.count).
    ///   - newTokens: the full token sequence for the new request.
    ///   - minReuse: minimum common-prefix length worth reusing (below this, prefilling fresh is
    ///     simpler and the savings are negligible).
    /// - Returns: a `Decision`. When `reuse` is true, trim `trimCount` tokens from the cache, then
    ///   feed `newTokens[reuseCount...]` as the model input.
    public static func plan(cachedTokens: [Int32], newTokens: [Int32], minReuse: Int = 8) -> Decision {
        guard !cachedTokens.isEmpty, !newTokens.isEmpty else { return .fresh }

        let common = commonPrefixLength(cachedTokens, newTokens)
        // Reuse only if the shared prefix is meaningful AND there is at least one new token to
        // feed (the iterator needs a non-empty input to produce the next token).
        guard common >= minReuse, common < newTokens.count else { return .fresh }

        let trim = cachedTokens.count - common
        return Decision(reuseCount: common, trimCount: trim, reuse: true)
    }

    public static func commonPrefixLength(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }
}
