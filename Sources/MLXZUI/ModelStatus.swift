import Foundation
import MLXZCore
import MLXZHub

/// The single, reconciled status of a model across the three sources that used to be
/// stitched per row (loaded from `ModelManager`, downloading from `DownloadManager`,
/// installed-on-disk from `LocalModelStore`, drafter from `DrafterPairing`). A card reads
/// exactly one of these to decide its primary affordance — so an already-installed search
/// hit shows "Load", never "Download".
public enum ModelStatus: Equatable, Sendable {
    case remote                        // not on disk, not downloading
    case downloading(fraction: Double)
    case downloadFailed
    case installed                     // on disk, not loaded
    case loading                       // being loaded into the engine
    case loaded                        // currently loaded
    case drafter(attached: Bool)       // a speculative companion — not directly loadable

    /// Pure precedence resolver (testable without any @MainActor state).
    /// loaded > loading > drafter > downloading > installed > downloadFailed > remote.
    public static func resolve(
        loaded: Bool,
        loading: Bool,
        isDrafter: Bool,
        attachedDrafter: Bool,
        downloadFraction: Double?,
        downloadFailed: Bool,
        installed: Bool
    ) -> ModelStatus {
        if loaded { return .loaded }
        if loading { return .loading }
        if isDrafter { return .drafter(attached: attachedDrafter) }
        if let f = downloadFraction { return .downloading(fraction: f) }
        if installed { return .installed }
        if downloadFailed { return .downloadFailed }
        return .remote
    }
}

/// Whether a model's weights comfortably fit in the machine's memory. Advisory only — it
/// warns before a user downloads/loads something their Mac can't hold, using a simple
/// fraction-of-physical-RAM rule (weights + KV cache + OS all share it).
public enum RAMFit: Equatable, Sendable {
    case fits
    case tight
    case exceeds
    case unknown

    public static func of(sizeBytes: Int64?, physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> RAMFit {
        guard let size = sizeBytes, size > 0, physicalMemory > 0 else { return .unknown }
        let ram = Double(physicalMemory)
        let s = Double(size)
        if s > ram * 0.80 { return .exceeds }   // weights alone eat most of RAM → won't run well
        if s > ram * 0.60 { return .tight }      // fits but leaves little headroom for KV/OS
        return .fits
    }
}

/// One byte formatter for the whole UI (replaces the two duplicated `byteString` helpers and
/// `CatalogEntry.sizeString`'s divergent GB/MB formatting).
public enum ByteFormat {
    public static func string(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
    public static func string(_ bytes: Int?) -> String? {
        guard let b = bytes, b > 0 else { return nil }
        return string(Int64(b))
    }
}

/// Humanize a large count for compact display: 142_000 → "142k", 1_200_000 → "1.2M".
public enum CountFormat {
    public static func compact(_ n: Int) -> String {
        switch n {
        case ..<1_000: return "\(n)"
        case ..<1_000_000:
            let k = Double(n) / 1_000
            return k < 10 ? String(format: "%.1fk", k) : String(format: "%.0fk", k)
        default:
            let m = Double(n) / 1_000_000
            return m < 10 ? String(format: "%.1fM", m) : String(format: "%.0fM", m)
        }
    }
}

/// A capability/quant facet a user can filter Discover results by. `all` matches everything.
public enum ModelFacet: String, CaseIterable, Sendable, Identifiable {
    case all = "All"
    case chat = "Chat"
    case vision = "Vision"
    case reasoning = "Reasoning"
    case code = "Code"
    case moe = "MoE"
    public var id: String { rawValue }
}

/// Pure predicates for the Discover list: hide auto-attaching drafter companions and apply
/// the selected capability facet. Kept dependency-light (Core types only) so it's testable.
public enum DiscoverFilter {
    /// Drafters (MTP/DSpark companions) are never directly loadable, so they're removed
    /// from top-level Discover results.
    public static func hidingDrafters(_ entries: [CatalogEntry]) -> [CatalogEntry] {
        entries.filter { !DrafterPairing.isDrafter($0.id) }
    }

    /// Whether an entry matches a capability facet.
    public static func matches(_ entry: CatalogEntry, facet: ModelFacet) -> Bool {
        switch facet {
        case .all: return true
        case .vision: return entry.isVision
        case .moe: return entry.isMoE
        case .reasoning: return ModelTypeInfo.tuning(repoID: entry.id) == .reasoning
        case .code: return ModelTypeInfo.tuning(repoID: entry.id) == .code
        case .chat:
            // "Chat" = a general instruct/chat model that isn't primarily vision/code/reasoning.
            let t = ModelTypeInfo.tuning(repoID: entry.id)
            return !entry.isVision && t != .code && t != .reasoning
        }
    }

    /// Apply drafter-hiding + facet filtering in one pass.
    public static func apply(_ entries: [CatalogEntry], facet: ModelFacet) -> [CatalogEntry] {
        hidingDrafters(entries).filter { matches($0, facet: facet) }
    }
}
