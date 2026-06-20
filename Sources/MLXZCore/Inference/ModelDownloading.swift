import Foundation

/// Downloads a model's files into the local cache without loading it into memory.
///
/// Separate from `ModelLoading` so the UI can pre-download (with progress/cancel) and load later.
/// Implemented in `MLXZInference` over the HuggingFace HubClient; injected at the composition root.
/// Progress of an in-flight download. `fraction` is 0…1; `completedBytes`/`totalBytes` are the byte
/// counts when the source reports them (0 when unknown).
public struct DownloadProgress: Sendable {
    public var fraction: Double
    public var completedBytes: Int64
    public var totalBytes: Int64
    public init(fraction: Double, completedBytes: Int64 = 0, totalBytes: Int64 = 0) {
        self.fraction = fraction
        self.completedBytes = completedBytes
        self.totalBytes = totalBytes
    }
}

public protocol ModelDownloading: Sendable {
    /// Download all files for `descriptor` into the HF cache. Calls `progress` ON THE MAIN ACTOR (so
    /// an `@Observable` UI model can update synchronously without an extra `Task` hop that SwiftUI may
    /// coalesce/drop). Honors task cancellation. Returns when present.
    func download(
        _ descriptor: ModelDescriptor,
        progress: @escaping @MainActor @Sendable (DownloadProgress) -> Void
    ) async throws
}
