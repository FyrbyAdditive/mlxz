import Foundation
import Observation
import MLXZCore

/// Tracks explicit model downloads (separate from loading), with progress and cancellation.
/// Drives an injected `ModelDownloading`; the concrete HubClient impl lives in MLXZInference.
@MainActor
@Observable
public final class DownloadManager {
    public struct Download: Identifiable, Sendable {
        public enum State: Sendable, Equatable {
            case downloading(fraction: Double, completedBytes: Int64, totalBytes: Int64)
            case done
            case failed(String)
            case cancelled
        }
        public let id: String          // repo id
        public var state: State
    }

    public private(set) var downloads: [Download] = []
    private var tasks: [String: Task<Void, Never>] = [:]
    private let downloader: any ModelDownloading

    public init(downloader: any ModelDownloading) {
        self.downloader = downloader
    }

    public func isDownloading(_ repoID: String) -> Bool {
        if case .downloading = downloads.first(where: { $0.id == repoID })?.state { return true }
        return false
    }

    public func start(_ repoID: String) {
        guard tasks[repoID] == nil else { return }
        upsert(repoID, .downloading(fraction: 0, completedBytes: 0, totalBytes: 0))

        let downloader = self.downloader
        let descriptor = ModelDescriptor(repoID: repoID)
        // @MainActor task: the actual network I/O runs off-main inside downloadSnapshot; we only need
        // the main actor for the state mutations. Progress fires on the main actor and updates
        // synchronously so SwiftUI reliably re-renders (the old detached `Task { @MainActor }` hop got
        // coalesced/dropped, which is why the bar only moved "sometimes").
        tasks[repoID] = Task { @MainActor [weak self] in
            do {
                try await downloader.download(descriptor) { @MainActor p in
                    self?.upsert(repoID, .downloading(
                        fraction: p.fraction, completedBytes: p.completedBytes, totalBytes: p.totalBytes))
                }
                self?.finish(repoID, .done)
            } catch is CancellationError {
                self?.finish(repoID, .cancelled)
            } catch {
                self?.finish(repoID, .failed(String(describing: error)))
            }
        }
    }

    public func cancel(_ repoID: String) {
        tasks[repoID]?.cancel()
    }

    /// Forget a download entry entirely (e.g. after the model is deleted from disk), so a stale
    /// `.failed`/`.cancelled`/`.done` state stops driving a "Retry"/"Downloaded" affordance in the UI.
    /// Cancels any in-flight task for the repo first.
    public func clear(_ repoID: String) {
        tasks[repoID]?.cancel()
        tasks[repoID] = nil
        downloads.removeAll { $0.id == repoID }
    }

    /// Drop terminal (`.failed`/`.cancelled`/`.done`) entries whose model is no longer on disk, as
    /// judged by `existsOnDisk`. A partial download that the user deleted leaves a stale `.failed`
    /// entry that would otherwise keep showing "Retry" forever in search results; this removes it so
    /// the row reverts to a plain "Download". Active downloads are never touched.
    public func pruneStale(existsOnDisk: (String) -> Bool) {
        downloads.removeAll { d in
            switch d.state {
            case .downloading: return false               // never prune an in-flight download
            case .done, .failed, .cancelled: return !existsOnDisk(d.id)
            }
        }
    }

    // MARK: - State

    private func upsert(_ id: String, _ state: Download.State) {
        if let idx = downloads.firstIndex(where: { $0.id == id }) {
            downloads[idx].state = state
        } else {
            downloads.append(Download(id: id, state: state))
        }
    }

    private func finish(_ id: String, _ state: Download.State) {
        upsert(id, state)
        tasks[id] = nil
    }
}
