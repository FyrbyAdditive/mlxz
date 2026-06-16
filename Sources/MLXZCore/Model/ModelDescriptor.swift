import Foundation

/// Identifies a model to load, independent of how it is stored or which engine loads it.
public struct ModelDescriptor: Sendable, Hashable, Codable, Identifiable {
    /// HuggingFace repo id, e.g. "mlx-community/Qwen3.6-35B-A3B-MTP-4bit".
    public var repoID: String
    /// Optional git revision / branch. Defaults to "main" when nil.
    public var revision: String?

    public var id: String { revision.map { "\(repoID)@\($0)" } ?? repoID }

    public init(repoID: String, revision: String? = nil) {
        self.repoID = repoID
        self.revision = revision
    }

    /// A short display name (last path component of the repo id).
    public var displayName: String {
        repoID.split(separator: "/").last.map(String.init) ?? repoID
    }
}
