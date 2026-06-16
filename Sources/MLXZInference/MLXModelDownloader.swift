import Foundation
import MLXZCore
import HuggingFace

/// Downloads model snapshots into the Python-compatible HuggingFace cache (the same cache the
/// loader reads from), so a pre-downloaded model loads instantly later.
public struct MLXModelDownloader: ModelDownloading {
    private let hub: HubClient

    public init(hub: HubClient = .default) {
        self.hub = hub
    }

    public func download(
        _ descriptor: ModelDescriptor,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        guard let repo = Repo.ID(rawValue: descriptor.repoID) else {
            throw APIError(kind: .invalidRequest, message: "Invalid repo id '\(descriptor.repoID)'", code: "invalid_repo_id")
        }
        _ = try await hub.downloadSnapshot(
            of: repo,
            revision: descriptor.revision ?? "main",
            progressHandler: { p in
                progress(p.fractionCompleted)
            }
        )
    }
}
