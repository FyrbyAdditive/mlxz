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
            case downloading(fraction: Double)
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
        upsert(repoID, .downloading(fraction: 0))

        let downloader = self.downloader
        let descriptor = ModelDescriptor(repoID: repoID)
        tasks[repoID] = Task { [weak self] in
            do {
                try await downloader.download(descriptor) { fraction in
                    Task { @MainActor in self?.upsert(repoID, .downloading(fraction: fraction)) }
                }
                await MainActor.run { self?.finish(repoID, .done) }
            } catch is CancellationError {
                await MainActor.run { self?.finish(repoID, .cancelled) }
            } catch {
                await MainActor.run { self?.finish(repoID, .failed(String(describing: error))) }
            }
        }
    }

    public func cancel(_ repoID: String) {
        tasks[repoID]?.cancel()
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
