import Foundation

/// Admits generation requests up to `maxConcurrent` in flight; further requests **wait** (FIFO)
/// for a slot rather than being rejected. VS Code Copilot fires concurrent requests (e.g.
/// fire-and-forget title generation alongside the main answer) and does NOT retry a 429, so
/// rejecting overlapping requests surfaced as a fatal "server is busy". Queuing is what every
/// mainstream local server does. A bounded wait queue (`maxWaiting`) sheds genuine overload with a
/// busy error; `maxWaiting <= 0` means unbounded.
///
/// With continuous batching, `maxConcurrent` can be > 1 (the batch engine decodes them together);
/// for the single-sequence MTP path it serializes (maxConcurrent 1) but waits instead of rejecting.
actor GenerationGate {
    private var active = 0
    private let defaultLimit: Int
    private let maxWaiting: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(maxConcurrent: Int = 1, maxWaiting: Int = 0) {
        self.defaultLimit = max(1, maxConcurrent)
        self.maxWaiting = maxWaiting
    }

    /// Acquire a slot, waiting (FIFO) if at capacity. The effective `limit` is supplied per call by
    /// the caller from the current engine's `maxConcurrency` — `1` serializes single-sequence /
    /// shared-prefix-cache models (so concurrent requests queue and can't clobber the cache), while
    /// a batchable model passes its `maxBatch` so requests reach the batch engine concurrently.
    /// Returns false only when the wait queue is bounded and full (genuine overload).
    func acquire(limit: Int? = nil) async -> Bool {
        let effectiveLimit = max(1, limit ?? defaultLimit)
        if active < effectiveLimit {
            active += 1
            return true
        }
        if maxWaiting > 0 && waiters.count >= maxWaiting {
            return false
        }
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            waiters.append(c)
        }
        // Resumed by `release()` which hands us the slot directly (active stays incremented).
        return true
    }

    func release() {
        if let next = waiters.first {
            waiters.removeFirst()
            next.resume()  // hand the slot directly to the next waiter (no decrement → no race)
        } else if active > 0 {
            active -= 1
        }
    }
}
