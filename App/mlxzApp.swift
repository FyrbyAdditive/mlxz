import SwiftUI
import MLXZUI
import MLXZInference

/// The macOS app. Composition root: supplies the concrete MLX loader to the UI's AppModel.
@main
struct MLXZApp: App {
    @State private var model: AppModel = {
        // Persisted perf settings, read PER LOAD by the loader's provider so GUI changes apply on the
        // next model load (KV-quant bits, prefix-cache slots, snapshot block).
        let perf = PerfSettings()
        return AppModel(
            loader: MLXModelLoader(perfProvider: { perf.engineOptions() }),
            downloader: MLXModelDownloader(),
            embeddingLoader: MLXEmbeddingLoader(),
            perfSettings: perf
        )
    }()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .frame(minWidth: 720, minHeight: 480)
        }
        .windowResizability(.contentMinSize)

        // Menu-bar presence so the server can keep running while windows are closed.
        MenuBarExtra("mlxz", systemImage: "brain") {
            MenuBarContent(model: model)
        }
        .menuBarExtraStyle(.menu)
    }
}

/// Minimal menu-bar controls.
struct MenuBarContent: View {
    @Bindable var model: AppModel

    var body: some View {
        if let descriptor = model.modelState.loadedDescriptor {
            Text("Model: \(descriptor.displayName)")
        } else {
            Text("No model loaded")
        }
        Text(model.serverRunning ? "Server: running" : "Server: stopped")
        Divider()
        if model.serverRunning {
            Button("Stop Server") { Task { await model.stopServer() } }
        } else {
            Button("Start Server") { Task { await model.startServer() } }
                .disabled(model.modelState.loadedDescriptor == nil)
        }
        Divider()
        Button("Quit mlxz") { NSApplication.shared.terminate(nil) }
    }
}
