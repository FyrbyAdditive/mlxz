import Foundation
import MLXLMCommon

/// Holds a persistent KV cache and the token ids it currently encodes, enabling prefix reuse
/// across requests. Non-Sendable (KVCache wraps MLXArray) — mutated only inside the owning
/// `ModelContainer`'s `perform`, where generation is already serialized by the GenerationGate.
///
/// Reference type so the engine (a struct) can carry and mutate it across calls.
/// `@unchecked Sendable`: only ever read/written inside the owning `ModelContainer.perform`,
/// which serializes access; never touched concurrently.
final class PromptCacheBox: @unchecked Sendable {
    /// The live KV cache, or nil before the first generation.
    var cache: [KVCache]?
    /// The token ids currently encoded by `cache` (prompt + previously generated tokens).
    var tokens: [Int32] = []

    /// For the MTP path: the persistent backbone + MTP-head caches and the exact token sequence
    /// they encode. The same `MTPCacheResult` instance is passed to every `mtpGenerate` call, which
    /// refills it on clean completion — enabling whole-prefix reuse across requests (the hybrid
    /// model's SSM layers preclude mid-sequence trimming, so only whole-prefix reuse is sound).
    let mtpResult = MTPCacheResult()

    /// Reset to empty (e.g. after model unload or when reuse is impossible).
    func reset() {
        cache = nil
        tokens = []
        mtpResult.snapshotModelCache = nil
        mtpResult.snapshotMtpCache = nil
        mtpResult.snapshotTokens = nil
        mtpResult.promptTokens = nil
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
