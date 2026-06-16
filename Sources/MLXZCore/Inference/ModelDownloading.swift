import Foundation

/// Downloads a model's files into the local cache without loading it into memory.
///
/// Separate from `ModelLoading` so the UI can pre-download (with progress/cancel) and load later.
/// Implemented in `MLXZInference` over the HuggingFace HubClient; injected at the composition root.
public protocol ModelDownloading: Sendable {
    /// Download all files for `descriptor` into the HF cache. Calls `progress` with 0…1 fractions.
    /// Honors task cancellation. Returns when the snapshot is fully present.
    func download(
        _ descriptor: ModelDescriptor,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws
}
