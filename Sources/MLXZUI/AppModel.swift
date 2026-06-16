import Foundation
import Observation
import Logging
import MLXZCore
import MLXZHub
import MLXZServer

/// The composition-root, main-actor state for the GUI. Holds the shared `ModelManager`,
/// `InferenceServer`, and `LogStore`, and exposes observable properties the SwiftUI views bind to.
///
/// The concrete MLX loader is injected (so this module stays free of MLX and builds under
/// `swift build`); the App target supplies `MLXModelLoader`.
@MainActor
@Observable
public final class AppModel {
    // Server config (bound to the UI).
    public var host: String = "127.0.0.1"
    public var port: Int = 8080
    public var bindLAN: Bool = false
    public var apiKey: String = ""

    // Observable state.
    public private(set) var modelState: ModelManager.State = .empty
    public private(set) var serverRunning: Bool = false

    public let logStore = LogStore()
    public let catalog = HubCatalog()
    public let localStore = LocalModelStore()
    public let downloads: DownloadManager

    private let manager: ModelManager
    private let server: InferenceServer
    private let logger = Logger(label: "mlxz.app")

    public init(loader: any ModelLoading, downloader: any ModelDownloading) {
        self.downloads = DownloadManager(downloader: downloader)
        let logStore = self.logStore
        let manager = ModelManager(loader: loader)
        self.manager = manager
        let store = self.localStore
        self.server = InferenceServer(
            manager: manager,
            logSink: { line in
                Task { @MainActor in logStore.append(line) }
            },
            extraModelIDs: { store.installedModels().map(\.descriptor.repoID) }
        )
    }

    /// Begin observing the manager's state stream. Call from a SwiftUI `.task {}`.
    public func observeModelState() async {
        for await state in await manager.states() {
            self.modelState = state
        }
    }

    // MARK: - Installed models

    public func installedModels() -> [InstalledModel] {
        localStore.installedModels()
    }

    // MARK: - Downloads

    public func startDownload(_ repoID: String) {
        logStore.append("Downloading \(repoID)…")
        downloads.start(repoID)
    }

    public func cancelDownload(_ repoID: String) {
        downloads.cancel(repoID)
    }

    // MARK: - Model lifecycle

    public func load(_ descriptor: ModelDescriptor) async {
        logStore.append("Loading \(descriptor.repoID)…")
        do {
            try await manager.load(descriptor)
            logStore.append("Loaded \(descriptor.repoID)")
        } catch {
            logStore.append("Load failed: \(error)")
        }
    }

    public func unload() async {
        await manager.unload()
        logStore.append("Unloaded model")
    }

    // MARK: - Server lifecycle

    public func startServer() async {
        let effectiveHost = bindLAN ? "0.0.0.0" : host
        let config = ServerConfig(
            host: effectiveHost,
            port: port,
            apiKey: apiKey.isEmpty ? nil : apiKey
        )
        do {
            try await server.start(config)
            serverRunning = true
        } catch {
            logStore.append("Server failed to start: \(error)")
        }
    }

    public func stopServer() async {
        await server.stop()
        serverRunning = false
    }

    // MARK: - Copilot config

    /// The VS Code chatLanguageModels.json entry for the currently-loaded model, if any.
    public func copilotConfigSnippet() -> String? {
        guard let descriptor = modelState.loadedDescriptor else { return nil }
        let caps = ModelCapabilityDetector.detect(repoID: descriptor.repoID)
        let effectiveHost = bindLAN ? host : "127.0.0.1"  // advertise loopback for the local case
        return CopilotConfig.modelEntry(
            repoID: descriptor.repoID,
            host: effectiveHost,
            port: port,
            capabilities: caps
        )
    }
}
