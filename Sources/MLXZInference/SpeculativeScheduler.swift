import Foundation
import MLX
import MLXLMCommon
import MLXZCore

/// The loaded DSpark drafter + its knobs, boxed to cross actor boundaries. The drafter is
/// a non-Sendable MLX `Module`; it is created and only ever touched inside the model's
/// `ModelContainer.perform` (same discipline as `PromptCacheBox`).
public final class DSparkRuntimeBox: @unchecked Sendable {
    let drafter: DSparkDrafter
    let blockCap: Int
    let confidenceThreshold: Float
    /// One adaptive draft on/off controller shared by all of this model's sessions —
    /// bootstrap/probe state persists across requests (mutated only inside the container).
    let adaptive = AdaptiveDraftController()

    init(drafter: DSparkDrafter, blockCap: Int, confidenceThreshold: Float) {
        self.drafter = drafter
        self.blockCap = blockCap
        self.confidenceThreshold = confidenceThreshold
    }
}

/// Fair, interleaving scheduler for speculative-decoding sessions — MTP (self-speculative)
/// or DSpark (standalone drafter). Speculative decode cannot batch — only one model forward
/// runs at a time — but without a scheduler a short request waits for a long request to
/// FULLY finish (a 10-token utility request could block ~7s behind a 200-token chat
/// answer). This advances each active request by ONE step round-robin, so a short request
/// finishes in ~its own compute time alongside a long one.
///
/// Correctness: each session's per-step math is identical to its single-flight loop.
/// Interleaving only changes the ORDER steps run on the (serialized) GPU, never a session's
/// own token sequence. All MLX work runs inside `container.perform`, so the model/caches
/// are never touched concurrently.
actor SpeculativeScheduler {
    /// Which kind of session this scheduler builds (fixed for the loaded model).
    enum Mode: Sendable {
        case mtp
        case dspark(DSparkRuntimeBox)
    }

    private let container: ModelContainer
    private let mode: Mode

    /// A submission waiting to be built into a session inside the container.
    private struct Pending {
        let input: SendableValueBox<UserInput>
        let parameters: GenerateParameters
        let box: PromptCacheBox?
        let continuation: AsyncStream<Generation>.Continuation
        let id: Int
        let reasoningBudget: Int
    }

    /// An admitted, stepping session plus where to store its prefix snapshots for reuse.
    private struct Active {
        let session: any SteppableSpeculativeSession
        let id: Int
        let lru: SnapshotLRU?
    }

    private var pending: [Pending] = []
    private var cancelled: Set<Int> = []
    private var nextID = 0
    private var running = false
    private let snapshotBlock: Int

    init(container: ModelContainer, snapshotBlock: Int = 512, mode: Mode = .mtp) {
        self.container = container
        self.snapshotBlock = max(1, snapshotBlock)
        self.mode = mode
    }

    /// Submit a request; returns its `Generation` stream. The scheduler starts on first submit.
    func submit(
        userInput: consuming UserInput,
        parameters: GenerateParameters,
        box: PromptCacheBox?,
        reasoningBudget: Int = 0
    ) -> AsyncStream<Generation> {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let id = nextID
        nextID += 1
        pending.append(
            Pending(
                input: SendableValueBox(userInput), parameters: parameters, box: box,
                continuation: continuation, id: id, reasoningBudget: reasoningBudget))
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

    /// The scheduler loop. Each tick performs one `container.perform` that admits new
    /// submissions (building their sessions) and advances every active session by one step.
    /// Snapshot inserts (cheap; arrays already materialized) happen back on the actor.
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

            // Fairness vs throughput: with >1 in flight, step each session ONCE per perform
            // so a short request interleaves promptly. When solo, run a small burst inside
            // one perform so the uncontended path isn't penalized by per-tick overhead.
            let contended = active.count + toAdmit.count > 1 || !cancelledNow.isEmpty
            let burst = contended ? 1 : 8
            let admitBox = SendableValueBox(toAdmit)
            let activeBox = SendableValueBox(active)
            let block = snapshotBlock
            let mode = self.mode
            let outBox = try? await container.perform { context in
                SendableValueBox(
                    await Self.tick(
                        context: context, mode: mode, admit: admitBox.consume(),
                        active: activeBox.consume(), cancelled: cancelledNow,
                        maxSteps: burst, snapshotBlock: block))
            }
            guard let outBox else { running = false; return }
            let out = outBox.consume()
            active = out.stillActive
            for ins in out.snapshots {
                ins.lru.insert(tokens: ins.tokens, modelCache: ins.model, mtpCache: ins.aux)
            }
            await Task.yield()
        }
    }

    /// Build one session for an admitted request (inside the container). nil when the
    /// loaded model can't serve this scheduler's mode (the caller finishes the stream).
    private static func buildSession(
        context: ModelContext, mode: Mode, p: Pending, lru: SnapshotLRU?, snapshotBlock: Int
    ) async -> (any SteppableSpeculativeSession)? {
        guard let lmInput = try? await context.processor.prepare(input: p.input.consume())
        else { return nil }
        let promptTokens = lmInput.text.tokens.reshaped([-1])
        let newTokens = promptTokens.asArray(Int32.self)

        let match = lru?.bestMatch(for: newTokens)
        let restoreCount = match?.tokens.count ?? 0

        // DIAGNOSTIC (env-gated MLXZ_PREFIX_DIAG=1): per-request reuse hit/miss + WHY.
        // `commonWithMRU` = longest shared prefix with the most-recent cached prompt.
        if ProcessInfo.processInfo.environment["MLXZ_PREFIX_DIAG"] == "1" {
            let mru = lru?.mostRecentTokens ?? []
            let common = MTPCacheReuse.commonPrefixLength(mru, newTokens)
            let head = newTokens.prefix(12).map(String.init).joined(separator: ",")
            FileHandle.standardError.write(Data(
                "[PREFIX] prompt=\(newTokens.count) reused=\(restoreCount) fresh=\(newTokens.count - restoreCount) commonWithMRU=\(common) mruLen=\(mru.count) head=[\(head)]\n".utf8))
        }

        let stopTokenIds = MTPStopTokens.build(
            eosTokenIds: context.configuration.eosTokenIds,
            tokenizerEOSTokenId: context.tokenizer.eosTokenId,
            extraEOSTokens: context.configuration.extraEOSTokens,
            tokenToId: { context.tokenizer.convertTokenToId($0) })

        switch mode {
        case .mtp:
            guard let model = context.model as? any MTPSpeculativeModel, model.hasMTP
            else { return nil }
            let restoreModel = match.map { $0.modelCache.map { $0.copy() } }
                ?? context.model.newCache(parameters: p.parameters)
            let restoreAux = match.map { $0.mtpCache.map { $0.copy() } }
                ?? model.makeMTPCache()
            let snapshotAt = max(restoreCount, newTokens.count - 256)
            return MTPSession(
                model: model, context: context, parameters: p.parameters,
                promptTokens: promptTokens, modelCache: restoreModel, mtpCache: restoreAux,
                skipPrefill: restoreCount, snapshotAt: snapshotAt,
                snapshotBlock: snapshotBlock,
                referenceTokens: lru?.mostRecentTokens ?? [],
                reasoningBudget: p.reasoningBudget,
                stopTokenIds: stopTokenIds, continuation: p.continuation,
                result: MTPCacheResult())

        case .dspark(let runtime):
            guard let model = context.model as? any DSparkTargetModel else { return nil }
            // Prefer the longest COMMON-PREFIX snapshot (whole-generation snapshots never
            // exact-prefix the next prompt — templates re-render prior turns — but both
            // cache kinds are trim-sound, so copy + trim back to the shared point).
            var partial: (model: [KVCache], ctx: [KVCacheSimple], common: Int)? = nil
            if let (entry, common) = lru?.bestCommonPrefix(for: newTokens), common > restoreCount {
                let m = entry.modelCache.map { $0.copy() }
                let c = entry.mtpCache.compactMap { $0.copy() as? KVCacheSimple }
                let overshoot = entry.tokens.count - common
                if overshoot > 0 {
                    trimPromptCache(m, numTokens: overshoot)
                    for cache in c { _ = cache.trim(overshoot) }
                }
                partial = (m, c, common)
            }
            if let partial {
                let sound = partial.ctx.allSatisfy { $0.offset == partial.common }
                    && (partial.model.first?.offset ?? -1) == partial.common
                if sound {
                    return DSparkSession(
                        model: model, drafter: runtime.drafter, context: context,
                        parameters: p.parameters, promptTokens: promptTokens,
                        modelCache: partial.model, ctxCaches: partial.ctx,
                        skipPrefill: partial.common, snapshotBlock: snapshotBlock,
                        referenceTokens: lru?.mostRecentTokens ?? [],
                        blockCap: runtime.blockCap,
                        confidenceThreshold: runtime.confidenceThreshold,
                        adaptiveController: runtime.adaptive,
                        reasoningBudget: p.reasoningBudget,
                        stopTokenIds: stopTokenIds, continuation: p.continuation,
                        result: MTPCacheResult())
                }
            }
            // The snapshot's aux slot carries the drafter's context caches (KVCacheSimple).
            // A snapshot whose aux doesn't restore cleanly would desync the drafter from
            // the target — fall back to a fresh prefill instead of decoding wrong context.
            var restoreModel = match.map { $0.modelCache.map { $0.copy() } }
            var restoreCtx = match.map { $0.mtpCache.compactMap { $0.copy() as? KVCacheSimple } }
            var skip = restoreCount
            let layerCount = runtime.drafter.makeContextCaches().count
            if let ctx = restoreCtx,
                !(ctx.count == layerCount && ctx.allSatisfy { $0.offset == restoreCount })
            {
                restoreModel = nil
                restoreCtx = nil
                skip = 0
            }
            return DSparkSession(
                model: model, drafter: runtime.drafter, context: context,
                parameters: p.parameters, promptTokens: promptTokens,
                modelCache: restoreModel ?? context.model.newCache(parameters: p.parameters),
                ctxCaches: restoreCtx ?? runtime.drafter.makeContextCaches(),
                skipPrefill: skip, snapshotBlock: snapshotBlock,
                referenceTokens: lru?.mostRecentTokens ?? [],
                blockCap: runtime.blockCap,
                confidenceThreshold: runtime.confidenceThreshold,
                adaptiveController: runtime.adaptive,
                reasoningBudget: p.reasoningBudget,
                stopTokenIds: stopTokenIds, continuation: p.continuation,
                result: MTPCacheResult())
        }
    }

    /// One tick (inside the container): admit new sessions, advance each active session up
    /// to `maxSteps` steps, collect survivors + captured snapshots.
    private static func tick(
        context: ModelContext, mode: Mode, admit: [Pending], active: [Active],
        cancelled: Set<Int>, maxSteps: Int, snapshotBlock: Int
    ) async -> TickResult {
        var sessions = active

        for p in admit {
            if cancelled.contains(p.id) { p.continuation.finish(); continue }
            let lru = p.box?.snapshotLRU
            guard
                let session = await buildSession(
                    context: context, mode: mode, p: p, lru: lru, snapshotBlock: snapshotBlock)
            else {
                p.continuation.finish()
                continue
            }
            sessions.append(Active(session: session, id: p.id, lru: lru))
        }

        var stillActive: [Active] = []
        var snapshots: [SnapshotInsert] = []
        for entry in sessions {
            if cancelled.contains(entry.id) { entry.session.cancel(); continue }
            var steps = 0
            while steps < maxSteps, !entry.session.isFinished {
                if entry.session.isPrefilling {
                    _ = entry.session.prefillStepOnce()
                } else {
                    _ = entry.session.decodeStepOnce()
                }
                if let snap = entry.session.takeSnapshot(), let lru = entry.lru {
                    snapshots.append(
                        SnapshotInsert(
                            lru: lru, tokens: snap.tokens, model: snap.model, aux: snap.aux))
                }
                steps += 1
            }
            if !entry.session.isFinished { stillActive.append(entry) }
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
    let aux: [KVCache]
}
