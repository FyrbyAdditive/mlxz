import Foundation

/// Serializes generation: MLX is memory-bound, so we run at most one generation at a time.
/// Requests beyond the queue depth are rejected with a busy error rather than blocking forever.
actor GenerationGate {
    private var active = 0
    private let maxConcurrent: Int

    init(maxConcurrent: Int = 1) {
        self.maxConcurrent = maxConcurrent
    }

    /// Try to acquire a slot. Returns false if the server is already at capacity.
    func tryAcquire() -> Bool {
        guard active < maxConcurrent else { return false }
        active += 1
        return true
    }

    func release() {
        if active > 0 { active -= 1 }
    }
}
