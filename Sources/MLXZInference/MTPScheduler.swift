import Foundation
import MLX
import MLXLMCommon
import MLXZCore

/// Fair, interleaving scheduler for MTP (self-speculative) decoding. MTP cannot batch — only one
/// model forward runs at a time — but without a scheduler a short request waits for a long request
/// to FULLY finish (the gate held the slot for the whole generation), so a 10-token utility request
/// could block ~7s behind a 200-token chat answer. This scheduler advances each active request by
/// ONE step round-robin, so a short request finishes in ~its own compute time alongside a long one.
///
/// Correctness: each `MTPSession`'s per-step math is byte-identical to the old single-flight
/// `mtpGenerate` loop. Interleaving only changes the ORDER steps run on the (serialized) GPU, never
/// a session's own token sequence (output is backbone-only; drafts are verified). All MLX work runs
/// inside `container.perform`, so the model/caches are never touched concurrently.
actor MTPScheduler {
    private let container: ModelContainer

    /// A submission waiting to be built into a session inside the container.
    private struct Pending {
        let input: SendableValueBox<UserInput>
        let parameters: GenerateParameters
        let box: PromptCacheBox?
        let continuation: AsyncStream<Generation>.Continuation
        let id: Int
    }

    /// An admitted, stepping session plus where to store its prefix snapshot for reuse.
    private struct Active {
        let session: MTPSession
        let id: Int
        let lru: SnapshotLRU?
    }

    private var pending: [Pending] = []
    private var cancelled: Set<Int> = []
    private var nextID = 0
    private var running = false

    init(container: ModelContainer) {
        self.container = container
    }

    /// Submit an MTP request; returns its `Generation` stream. The scheduler starts on first submit.
    func submit(
        userInput: consuming UserInput,
        parameters: GenerateParameters,
        box: PromptCacheBox?
    ) -> AsyncStream<Generation> {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let id = nextID
        nextID += 1
        pending.append(
            Pending(
                input: SendableValueBox(userInput), parameters: parameters, box: box,
                continuation: continuation, id: id))
        continuation.onTermination = { [weak self] _ in
            Task { await self?.markCancelled(id) }
        }
        if !running {
            running = true
            Task { await self.run() }
        }
        return stream
    }

    private func markCancelled(_ id: Int) { cancelled.insert(id) }

    /// The scheduler loop. Each tick performs one `container.perform` that admits new submissions
    /// (building their sessions) and advances every active session by one step. Snapshot inserts
    /// (cheap; arrays already materialized) happen back on the actor between ticks.
    private func run() async {
        var active: [Active] = []

        while true {
            let toAdmit = pending
            pending = []
            let cancelledNow = cancelled

            if active.isEmpty && toAdmit.isEmpty {
                running = false
                return
            }

            // Fairness vs throughput: with >1 in flight, step each session ONCE per perform so a
            // short request interleaves promptly. When solo (one active, nothing pending/cancelled),
            // run a small burst inside one perform so a newcomer still interrupts within `burst`
            // steps but the uncontended path isn't penalized by per-tick overhead.
            let contended = active.count + toAdmit.count > 1 || !cancelledNow.isEmpty
            let burst = contended ? 1 : 8
            let admitBox = SendableValueBox(toAdmit)
            let activeBox = SendableValueBox(active)
            let outBox = try? await container.perform { context in
                SendableValueBox(
                    await Self.tick(
                        context: context, admit: admitBox.consume(), active: activeBox.consume(),
                        cancelled: cancelledNow, maxSteps: burst))
            }
            guard let outBox else { running = false; return }
            let out = outBox.consume()
            active = out.stillActive
            for ins in out.snapshots {
                ins.lru.insert(tokens: ins.tokens, modelCache: ins.model, mtpCache: ins.mtp)
            }
            await Task.yield()
        }
    }

    /// One tick (inside the container): build/admit new sessions, advance each active session one
    /// step, collect survivors + any captured snapshots. `static` + boxed I/O so it runs in the
    /// `@Sendable` perform closure without actor-isolation conflicts.
    private static func tick(
        context: ModelContext, admit: [Pending], active: [Active], cancelled: Set<Int>,
        maxSteps: Int
    ) async -> TickResult {
        guard let model = context.model as? any MTPSpeculativeModel, model.hasMTP else {
            for a in active { a.session.cancel() }
            for p in admit { p.continuation.finish() }
            return TickResult(stillActive: [], snapshots: [])
        }

        var sessions = active

        // Admit new submissions as sessions.
        for p in admit {
            if cancelled.contains(p.id) { p.continuation.finish(); continue }
            guard let lmInput = try? await context.processor.prepare(input: p.input.consume()) else {
                p.continuation.finish(); continue
            }
            let promptTokens = lmInput.text.tokens.reshaped([-1])
            let newTokens = promptTokens.asArray(Int32.self)
            let lru = p.box?.snapshotLRU

            let restoreModel: [KVCache]
            let restoreMtp: [KVCache]
            var restoreCount = 0
            if let match = lru?.bestMatch(for: newTokens) {
                restoreModel = match.modelCache.map { $0.copy() }
                restoreMtp = match.mtpCache.map { $0.copy() }
                restoreCount = match.tokens.count
            } else {
                restoreModel = context.model.newCache(parameters: p.parameters)
                restoreMtp = model.makeMTPCache()
            }
            let snapshotAt = max(restoreCount, newTokens.count - 256)

            let stopTokenIds = MTPStopTokens.build(
                eosTokenIds: context.configuration.eosTokenIds,
                tokenizerEOSTokenId: context.tokenizer.eosTokenId,
                extraEOSTokens: context.configuration.extraEOSTokens,
                tokenToId: { context.tokenizer.convertTokenToId($0) })

            let session = MTPSession(
                model: model, context: context, parameters: p.parameters,
                promptTokens: promptTokens, modelCache: restoreModel, mtpCache: restoreMtp,
                skipPrefill: restoreCount, snapshotAt: snapshotAt,
                stopTokenIds: stopTokenIds, continuation: p.continuation, result: MTPCacheResult())
            sessions.append(Active(session: session, id: p.id, lru: lru))
        }

        // Advance each session up to `maxSteps` steps (1 when contended — strict fairness; a burst
        // when solo — amortizes per-tick overhead). Collect survivors + captured snapshots.
        var stillActive: [Active] = []
        var snapshots: [SnapshotInsert] = []
        for entry in sessions {
            if cancelled.contains(entry.id) { entry.session.cancel(); continue }
            var steps = 0
            while steps < maxSteps, entry.session.phase != .finished {
                switch entry.session.phase {
                case .prefilling: _ = entry.session.prefillStepOnce()
                case .decoding: _ = entry.session.decodeStepOnce()
                case .finished: break
                }
                if let snap = entry.session.takeCapturedSnapshot(), let lru = entry.lru {
                    snapshots.append(
                        SnapshotInsert(
                            lru: lru, tokens: snap.tokens, model: snap.model, mtp: snap.mtp))
                }
                steps += 1
            }
            if entry.session.phase != .finished { stillActive.append(entry) }
        }

        return TickResult(stillActive: stillActive, snapshots: snapshots)
    }

    private struct TickResult {
        let stillActive: [Active]
        let snapshots: [SnapshotInsert]
    }
}

/// A captured prefix snapshot to insert into an LRU after a session reaches its snapshot point.
struct SnapshotInsert {
    let lru: SnapshotLRU
    let tokens: [Int32]
    let model: [KVCache]
    let mtp: [KVCache]
}
