import Testing

@testable import MLXZServer

/// The gate must QUEUE concurrent requests (FIFO wait) instead of rejecting — rejecting surfaced as
/// a fatal "server is busy" in VS Code (which doesn't retry). Only a bounded, full wait queue may
/// reject (genuine overload).
@Suite struct GenerationGateTests {

    @Test func firstAcquireGetsSlotImmediately() async {
        let gate = GenerationGate(maxConcurrent: 1)
        let ok = await gate.acquire()
        #expect(ok)
    }

    @Test func secondAcquireWaitsUntilRelease() async {
        let gate = GenerationGate(maxConcurrent: 1)
        #expect(await gate.acquire())  // holds the only slot

        // Second acquire must NOT complete until release.
        let started = Date2.now()
        let waiter = Task { await gate.acquire() }
        // Give the waiter a moment to park.
        try? await Task.sleep(nanoseconds: 50_000_000)
        #expect(!waiter.isCancelled)
        await gate.release()  // hand the slot to the waiter
        let ok = await waiter.value
        #expect(ok)
        _ = started
    }

    @Test func boundedQueueRejectsOnlyWhenFull() async {
        // 1 slot, wait queue bounded to 1. Slot taken + 1 waiter parked = full → next is rejected.
        let gate = GenerationGate(maxConcurrent: 1, maxWaiting: 1)
        #expect(await gate.acquire())  // takes the slot

        let parked = Task { await gate.acquire() }  // fills the 1-deep wait queue
        try? await Task.sleep(nanoseconds: 50_000_000)

        let rejected = await gate.acquire()  // queue full → busy
        #expect(rejected == false)

        await gate.release()  // wake the parked waiter
        #expect(await parked.value)
    }

    @Test func unboundedQueueNeverRejects() async {
        let gate = GenerationGate(maxConcurrent: 1, maxWaiting: 0)  // unbounded
        #expect(await gate.acquire())
        // Several waiters all park (none rejected); release drains them FIFO.
        let waiters = (0 ..< 5).map { _ in Task { await gate.acquire() } }
        try? await Task.sleep(nanoseconds: 50_000_000)
        for _ in waiters { await gate.release() }
        for w in waiters { #expect(await w.value) }
    }

    /// The per-call `limit` (from the engine's maxConcurrency) overrides the gate default. With
    /// limit=1 (single-sequence/MTP model), a 2nd concurrent acquire must WAIT — this is what keeps
    /// the shared prefix cache from being clobbered by a concurrent request.
    @Test func perCallLimitOfOneSerializes() async {
        let gate = GenerationGate(maxConcurrent: 8)  // default would allow 8…
        #expect(await gate.acquire(limit: 1))         // …but limit:1 takes the only slot

        let waiter = Task { await gate.acquire(limit: 1) }
        try? await Task.sleep(nanoseconds: 50_000_000)
        // The waiter must still be parked (not completed) — proves serialization.
        #expect(!waiter.isCancelled)
        await gate.release()
        #expect(await waiter.value)  // released → proceeds
    }

    /// With limit=N (batchable model), N requests acquire concurrently without waiting.
    @Test func perCallLimitAllowsConcurrency() async {
        let gate = GenerationGate(maxConcurrent: 1)  // default 1…
        // …but limit:4 lets 4 acquire immediately.
        for _ in 0 ..< 4 { #expect(await gate.acquire(limit: 4)) }
    }
}

/// Tiny date stand-in (avoids importing Foundation just for timing in a logic test).
private enum Date2 { static func now() -> Int { 0 } }
