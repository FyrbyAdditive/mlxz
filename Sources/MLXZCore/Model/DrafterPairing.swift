import Foundation

/// Relates an MTP *drafter* checkpoint to its *base* model. mlx-community publishes drafters as
/// `<base>-MTP-<quant>` (e.g. `mlx-community/Qwen3.6-27B-MTP-4bit` pairs with
/// `mlx-community/Qwen3.6-27B-4bit`). A drafter only holds the speculative head and must be
/// attached to its base; this derives the pairing both ways so the UI can group and auto-attach them.
public enum DrafterPairing {
    /// True if `repoID` names a standalone MTP drafter.
    public static func isDrafter(_ repoID: String) -> Bool {
        baseRepoID(forDrafter: repoID) != nil
    }

    /// The base model repo id for a drafter, or nil if `repoID` isn't a drafter.
    /// `org/Foo-MTP-4bit` → `org/Foo-4bit`; `org/Foo-MTP` → `org/Foo`.
    public static func baseRepoID(forDrafter repoID: String) -> String? {
        let (prefix, name) = split(repoID)
        // Match `-MTP-<suffix>` (e.g. -MTP-4bit) or a trailing `-MTP`, case-insensitive.
        guard let range = name.range(of: "-MTP", options: [.caseInsensitive]) else { return nil }
        // Only treat as a drafter if "-MTP" is followed by end or a quant-like suffix ("-...").
        let after = name[range.upperBound...]
        guard after.isEmpty || after.hasPrefix("-") else { return nil }
        let base = String(name[..<range.lowerBound]) + String(after)
        return prefix.map { "\($0)/\(base)" } ?? base
    }

    /// The drafter repo id matching a base model, by inserting `-MTP` before the quant suffix.
    /// `org/Foo-4bit` → `org/Foo-MTP-4bit`. Used to look one up among installed models.
    public static func drafterRepoID(forBase repoID: String) -> String {
        let (prefix, name) = split(repoID)
        let quantSuffixes = ["-2bit", "-3bit", "-4bit", "-5bit", "-6bit", "-8bit", "-bf16", "-fp16"]
        for q in quantSuffixes where name.lowercased().hasSuffix(q) {
            let stem = String(name.dropLast(q.count))
            let drafter = "\(stem)-MTP\(q)"
            return prefix.map { "\($0)/\(drafter)" } ?? drafter
        }
        let drafter = "\(name)-MTP"
        return prefix.map { "\($0)/\(drafter)" } ?? drafter
    }

    private static func split(_ repoID: String) -> (org: String?, name: String) {
        if let slash = repoID.lastIndex(of: "/") {
            return (String(repoID[..<slash]), String(repoID[repoID.index(after: slash)...]))
        }
        return (nil, repoID)
    }
}
