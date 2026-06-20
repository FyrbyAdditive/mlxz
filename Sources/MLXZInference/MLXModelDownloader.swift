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
        progress: @escaping @MainActor @Sendable (DownloadProgress) -> Void
    ) async throws {
        guard let repo = Repo.ID(rawValue: descriptor.repoID) else {
            throw APIError(kind: .invalidRequest, message: "Invalid repo id '\(descriptor.repoID)'", code: "invalid_repo_id")
        }

        // swift-huggingface's aggregate `Progress` does NOT report a large file's in-flight bytes:
        // its per-file child progress is updated by the URLSession delegate, but the parent aggregation
        // is broken — the handler flip-flops between "small files done" (≈3%) and a bogus 100% while the
        // one big safetensors file downloads, never the bytes in between (reproduced with AND without
        // Xet). So we DON'T trust the library's bytes for progress; we only mine its callback for the
        // accurate *total* size. Completed bytes come from polling disk + the in-flight URLSession temp
        // file, which (verified) grows monotonically with the actual download.
        let cacheDir = Self.modelCacheDirectory(for: repo)
        let total = TotalBox()
        let lastReported = TotalBox()  // monotonic floor so the bar never jumps backwards

        // Background poller every ~250ms: completed = on-disk cache bytes (finished files) + the
        // actively-growing CFNetworkDownload temp (the big file in flight). The big file is moved into
        // the cache atomically on completion, so disk and in-flight are disjoint — summing them is the
        // true total downloaded so far.
        let poller = Task { @MainActor in
            while !Task.isCancelled {
                let tot = total.value
                if tot > 0 {
                    let raw = Self.directorySize(cacheDir) + Self.inFlightDownloadBytes()
                    let completed = min(max(raw, lastReported.value), tot)
                    lastReported.value = completed
                    progress(DownloadProgress(
                        fraction: Double(completed) / Double(tot),
                        completedBytes: completed, totalBytes: tot))
                }
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
        defer { poller.cancel() }

        _ = try await hub.downloadSnapshot(
            of: repo,
            revision: descriptor.revision ?? "main",
            progressHandler: { @MainActor p in
                // Mine ONLY the total from the library (it's accurate); ignore its broken byte count.
                let tot = p.totalUnitCount
                if tot > 0 { total.value = tot }
            }
        )
        // Final 100% (the move-into-place may complete just after the last poll).
        if total.value > 0 {
            await MainActor.run {
                progress(DownloadProgress(fraction: 1, completedBytes: total.value, totalBytes: total.value))
            }
        }
    }

    /// The `models--org--name` cache directory for a repo, under the first HF cache root.
    static func modelCacheDirectory(for repo: Repo.ID) -> URL {
        let root = hubCacheRoot()
        let dirName = "models--" + repo.description.replacingOccurrences(of: "/", with: "--")
        return root.appendingPathComponent(dirName)
    }

    /// The active HF hub cache root (`HF_HUB_CACHE` → `$HF_HOME/hub` → `~/.cache/huggingface/hub`).
    static func hubCacheRoot() -> URL {
        let env = ProcessInfo.processInfo.environment
        if let c = env["HF_HUB_CACHE"], !c.isEmpty { return URL(fileURLWithPath: c) }
        if let h = env["HF_HOME"], !h.isEmpty {
            return URL(fileURLWithPath: h).appendingPathComponent("hub")
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/huggingface/hub")
    }

    /// Sum of bytes in *actively-growing* URLSession download temp files. `URLSession.download`
    /// streams each file to a `CFNetworkDownload_*.tmp` in the system temp dir and atomically MOVES it
    /// into the cache's `blobs/` only on completion — so for a large model file these temps are the
    /// ONLY place the in-flight bytes are observable (verified: the cache dir shows nothing until the
    /// file lands, then jumps 278MB at once).
    ///
    /// We must ignore STALE temps left by earlier/interrupted downloads (which can be gigabytes and
    /// would otherwise swamp the count and peg the bar at 100%): only count files modified within the
    /// last few seconds, i.e. ones the current download is still writing.
    static func inFlightDownloadBytes(activeWithin seconds: TimeInterval = 4) -> Int64 {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]
        guard let items = try? fm.contentsOfDirectory(at: tmp, includingPropertiesForKeys: keys) else {
            return 0
        }
        let cutoff = Date(timeIntervalSinceNow: -seconds)
        var total: Int64 = 0
        for url in items where url.lastPathComponent.hasPrefix("CFNetworkDownload_") {
            guard let vals = try? url.resourceValues(forKeys: Set(keys)),
                  let mtime = vals.contentModificationDate, mtime >= cutoff,
                  let size = vals.fileSize else { continue }
            total += Int64(size)
        }
        return total
    }

    /// Recursive byte size of a directory (resolves symlinks into the shared `blobs/`). 0 if absent.
    static func directorySize(_ dir: URL) -> Int64 {
        let fm = FileManager.default
        guard let en = fm.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else {
            return 0
        }
        var total: Int64 = 0
        for case let url as URL in en {
            let vals = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if vals?.isRegularFile == true, let s = vals?.fileSize { total += Int64(s) }
        }
        return total
    }

    /// Thread-safe box for the total byte count (written by the library callback, read by the poller).
    private final class TotalBox: @unchecked Sendable {
        private let lock = NSLock()
        private var _value: Int64 = 0
        var value: Int64 {
            get { lock.lock(); defer { lock.unlock() }; return _value }
            set { lock.lock(); _value = newValue; lock.unlock() }
        }
    }
}
