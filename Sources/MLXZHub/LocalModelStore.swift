import Foundation
import MLXZCore

/// A model already present in the local HuggingFace cache.
public struct InstalledModel: Sendable, Identifiable, Hashable {
    public var descriptor: ModelDescriptor
    public var directory: URL
    public var sizeBytes: Int64
    public var modelType: String?
    /// True when config.json declares vision components (multimodal model).
    public var hasVisionConfig: Bool = false

    public var id: String { descriptor.id }
    public var displayName: String { descriptor.displayName }

    public var capabilities: ModelCapabilities {
        ModelCapabilityDetector.detect(
            repoID: descriptor.repoID, modelType: modelType, hasVisionConfig: hasVisionConfig)
    }
}

/// Enumerates models already downloaded into the HuggingFace cache, with no network access.
///
/// The swift-huggingface `HubClient` uses a Python-compatible cache laid out as
/// `<root>/models--<org>--<name>/snapshots/<revision>/…`. We scan for snapshot directories
/// containing a `config.json`, which marks a usable model.
public struct LocalModelStore: Sendable {
    private let cacheRoots: [URL]

    public init(cacheRoots: [URL]? = nil) {
        self.cacheRoots = cacheRoots ?? Self.defaultCacheRoots()
    }

    /// Default HuggingFace cache locations checked, in order.
    public static func defaultCacheRoots() -> [URL] {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser
        var roots: [URL] = []
        if let env = ProcessInfo.processInfo.environment["HF_HOME"] {
            roots.append(URL(fileURLWithPath: env).appendingPathComponent("hub"))
        }
        roots.append(home.appendingPathComponent(".cache/huggingface/hub"))
        roots.append(home.appendingPathComponent("Documents/huggingface/models"))
        return roots
    }

    /// List installed models across all cache roots.
    public func installedModels() -> [InstalledModel] {
        var results: [InstalledModel] = []
        let fm = FileManager.default
        for root in cacheRoots {
            guard let entries = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else { continue }
            for entry in entries where entry.lastPathComponent.hasPrefix("models--") {
                if let model = Self.installedModel(at: entry) {
                    results.append(model)
                }
            }
        }
        // De-duplicate by repo id, keeping the first (highest-priority root).
        var seen = Set<String>()
        return results.filter { seen.insert($0.descriptor.repoID).inserted }
    }

    /// Whether *any* cache directory exists on disk for a repo id — including a PARTIAL download that
    /// has no `config.json` yet (so it wouldn't show up in `installedModels()`). Used to decide whether
    /// a prior failed download still has bytes worth retrying, vs. one whose files have been deleted.
    public func hasCacheDirectory(forRepoID repoID: String) -> Bool {
        let fm = FileManager.default
        let dirName = "models--" + repoID.replacingOccurrences(of: "/", with: "--")
        for root in cacheRoots {
            let dir = root.appendingPathComponent(dirName)
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue { return true }
        }
        return false
    }

    /// Delete an installed model from the HF cache. `model.directory` is the snapshot revision dir
    /// (`models--org--name/snapshots/<rev>`); we remove the whole `models--org--name` root (snapshots
    /// + the shared `blobs/`) so no orphaned files remain. Returns true on success.
    @discardableResult
    public func delete(_ model: InstalledModel) -> Bool {
        let fm = FileManager.default
        // revision -> snapshots -> models--org--name
        let modelRoot = model.directory.deletingLastPathComponent().deletingLastPathComponent()
        // Safety: only delete a directory that is actually an HF model cache entry.
        guard modelRoot.lastPathComponent.hasPrefix("models--") else { return false }
        do {
            try fm.removeItem(at: modelRoot)
            return true
        } catch {
            return false
        }
    }

    /// Parse a single `models--org--name` cache directory into an `InstalledModel`, if it holds
    /// a snapshot with a `config.json`.
    static func installedModel(at modelDir: URL) -> InstalledModel? {
        let fm = FileManager.default
        // "models--mlx-community--Qwen3.6-4B-4bit" → "mlx-community/Qwen3.6-4B-4bit"
        let name = modelDir.lastPathComponent
        guard name.hasPrefix("models--") else { return nil }
        let repoID = name.dropFirst("models--".count).replacingOccurrences(of: "--", with: "/")

        let snapshots = modelDir.appendingPathComponent("snapshots")
        guard let revisions = try? fm.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil) else {
            return nil
        }
        // Pick the snapshot that actually contains a config.json.
        for revision in revisions {
            let config = revision.appendingPathComponent("config.json")
            guard fm.fileExists(atPath: config.path) else { continue }
            let info = Self.readConfig(config)
            let size = Self.directorySize(revision)
            return InstalledModel(
                descriptor: ModelDescriptor(repoID: repoID),
                directory: revision,
                sizeBytes: size,
                modelType: info.modelType,
                hasVisionConfig: info.hasVision
            )
        }
        return nil
    }

    /// Read `model_type` and a vision signal from config.json. A model is multimodal if config
    /// declares any vision component (`vision_config`, an image token, etc.) — repo names often
    /// omit a `-vl` marker, so this is the reliable source.
    static func readConfig(_ configURL: URL) -> (modelType: String?, hasVision: Bool) {
        guard let data = try? Data(contentsOf: configURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return (nil, false) }
        let visionKeys = ["vision_config", "image_token_id", "vision_start_token_id", "vision_tower"]
        let hasVision = visionKeys.contains { obj[$0] != nil }
        return (obj["model_type"] as? String, hasVision)
    }

    /// Best-effort recursive size of a directory (follows the snapshot's symlinks into blobs).
    /// On-disk size of a snapshot directory. HF caches store snapshot entries as symlinks into a
    /// shared `blobs/` directory, so we resolve each symlink to its real file and sum the unique
    /// blobs (de-duplicating blobs shared across files/revisions). The default `.fileSizeKey`
    /// reports 0 for symlinks, which is why models showed as 0 KB.
    static func directorySize(_ url: URL) -> Int64 {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: url,
                includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey])
        else { return 0 }

        var total: Int64 = 0
        var countedBlobs = Set<String>()
        for case let fileURL as URL in enumerator {
            // Resolve symlinks to the real (blob) file; measure that.
            let resolved = fileURL.resolvingSymlinksInPath()
            let path = resolved.path
            guard countedBlobs.insert(path).inserted else { continue }  // skip already-counted blobs
            let values = try? resolved.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true, let size = values?.fileSize {
                total += Int64(size)
            }
        }
        return total
    }
}
