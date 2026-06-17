import SwiftUI
import MLXZCore
import MLXZHub

/// Browse installed models, search the HuggingFace catalog, and load a model.
struct ModelLibraryView: View {
    @Bindable var model: AppModel

    @State private var searchText = ""
    @State private var results: [CatalogEntry] = []
    @State private var installed: [InstalledModel] = []
    @State private var isSearching = false
    @State private var pasteRepoID = ""

    var body: some View {
        Form {
            Section("Load by repo id") {
                HStack {
                    TextField("e.g. mlx-community/Qwen3.6-35B-A3B-MTP-4bit", text: $pasteRepoID)
                        .textFieldStyle(.roundedBorder)
                    Button("Load") {
                        let id = pasteRepoID.trimmingCharacters(in: .whitespaces)
                        guard !id.isEmpty else { return }
                        Task { await model.load(ModelDescriptor(repoID: id)) }
                    }
                    .disabled(pasteRepoID.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }

            Section("Installed") {
                if installed.isEmpty {
                    Text("No models downloaded yet.").foregroundStyle(.secondary)
                }
                ForEach(installed) { m in
                    HStack {
                        VStack(alignment: .leading) {
                            HStack(spacing: 6) {
                                Text(m.displayName).font(.body)
                                if isLoaded(m.descriptor.repoID) { loadedBadge }
                            }
                            Text(byteString(m.sizeBytes)).font(.caption).foregroundStyle(.secondary)
                        }
                        CapabilityBadges(capabilities: m.capabilities)
                        Spacer()
                        loadControl(for: m.descriptor.repoID)
                    }
                }
            }

            Section("Search HuggingFace (MLX)") {
                HStack {
                    TextField("Search models…", text: $searchText)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { runSearch() }
                    Button("Search") { runSearch() }.disabled(isSearching)
                }
                if isSearching { ProgressView() }
                ForEach(results) { entry in
                    HStack {
                        VStack(alignment: .leading) {
                            HStack(spacing: 6) {
                                Text(entry.displayName)
                                if isLoaded(entry.id) { loadedBadge }
                            }
                            Text("↓ \(entry.downloads)  ·  \(entry.quantization ?? "—")")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        CapabilityBadges(capabilities: entry.capabilities)
                        Spacer()
                        downloadControl(for: entry.id)
                        loadControl(for: entry.id)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Models")
        .task { installed = model.installedModels() }
    }

    /// Whether `repoID` is the currently-loaded model.
    private func isLoaded(_ repoID: String) -> Bool {
        model.modelState.loadedDescriptor?.repoID == repoID
    }

    /// Whether `repoID` is currently being loaded.
    private func isLoading(_ repoID: String) -> Bool {
        if case .loading(let d, _) = model.modelState { return d.repoID == repoID }
        return false
    }

    private var loadedBadge: some View {
        Text("Loaded")
            .font(.caption2)
            .padding(.horizontal, 5).padding(.vertical, 2)
            .background(Color.green.opacity(0.18), in: Capsule())
            .foregroundStyle(.green)
    }

    /// Load / Loading… / Unload control reflecting the model lifecycle state.
    @ViewBuilder
    private func loadControl(for repoID: String) -> some View {
        if isLoaded(repoID) {
            Button("Unload", role: .destructive) { Task { await model.unload() } }
        } else if isLoading(repoID) {
            HStack(spacing: 4) {
                ProgressView().controlSize(.small)
                Text("Loading…").font(.caption).foregroundStyle(.secondary)
            }
        } else {
            Button("Load") { Task { await model.load(ModelDescriptor(repoID: repoID)) } }
        }
    }

    /// A Download / progress / cancel control reflecting the DownloadManager state.
    @ViewBuilder
    private func downloadControl(for repoID: String) -> some View {
        let download = model.downloads.downloads.first { $0.id == repoID }
        switch download?.state {
        case .downloading(let fraction):
            HStack(spacing: 4) {
                ProgressView(value: fraction).frame(width: 60)
                Button("✕") { model.cancelDownload(repoID) }.buttonStyle(.borderless)
            }
        case .done:
            Label("Downloaded", systemImage: "checkmark.circle").labelStyle(.iconOnly).foregroundStyle(.green)
        case .failed:
            Button("Retry") { model.startDownload(repoID) }.foregroundStyle(.red)
        default:
            Button("Download") { model.startDownload(repoID) }
        }
    }

    private func runSearch() {
        isSearching = true
        Task {
            defer { isSearching = false }
            results = (try? await model.catalog.search(query: searchText)) ?? []
        }
    }

    private func byteString(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

/// Small capability chips (tools / vision / MTP).
struct CapabilityBadges: View {
    let capabilities: ModelCapabilities
    var body: some View {
        HStack(spacing: 4) {
            if capabilities.contains(.vision) { badge("vision", .purple) }
            if capabilities.contains(.speculative) { badge("MTP", .orange) }
            if capabilities.contains(.tools) { badge("tools", .blue) }
        }
    }
    private func badge(_ text: String, _ color: Color) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 5).padding(.vertical, 2)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }
}
