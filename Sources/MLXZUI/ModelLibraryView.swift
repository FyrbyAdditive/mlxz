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
                ForEach(baseModels) { m in
                    installedRow(m, indented: false)
                    // Nest a matching installed drafter directly under its base model.
                    if let drafter = installedDrafter(forBase: m.descriptor.repoID) {
                        installedRow(drafter, indented: true)
                    }
                }
                // Orphan drafters whose base model isn't installed (shown so they're not hidden).
                ForEach(orphanDrafters) { d in
                    installedRow(d, indented: false)
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
                            Text(entry.displayName)
                            Text("↓ \(entry.downloads)  ·  \(entry.quantization ?? "—")")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        CapabilityBadges(capabilities: entry.capabilities, loaded: isLoaded(entry.id))
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

    private func isDrafter(_ repoID: String) -> Bool { DrafterPairing.isDrafter(repoID) }

    /// Whether `repoID` is the MTP drafter attached to the currently-loaded model.
    private func isAttachedDrafter(_ repoID: String) -> Bool {
        model.modelState.attachedDrafterID == repoID
    }

    // MARK: Installed-list grouping

    /// Installed models that are NOT drafters (base / standalone models).
    private var baseModels: [InstalledModel] {
        installed.filter { !isDrafter($0.descriptor.repoID) }
    }

    /// The installed drafter that pairs with `baseRepoID`, if present.
    private func installedDrafter(forBase baseRepoID: String) -> InstalledModel? {
        let expected = DrafterPairing.drafterRepoID(forBase: baseRepoID)
        return installed.first { $0.descriptor.repoID == expected }
    }

    /// Drafters whose base model is not installed — shown standalone so they aren't hidden.
    private var orphanDrafters: [InstalledModel] {
        let baseIDs = Set(baseModels.map(\.descriptor.repoID))
        return installed.filter {
            isDrafter($0.descriptor.repoID)
                && DrafterPairing.baseRepoID(forDrafter: $0.descriptor.repoID).map {
                    !baseIDs.contains($0)
                } ?? true
        }
    }

    /// One installed-model row; `indented` nests a drafter under its base model.
    @ViewBuilder
    private func installedRow(_ m: InstalledModel, indented: Bool) -> some View {
        HStack {
            if indented {
                Image(systemName: "arrow.turn.down.right")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading) {
                Text(m.displayName).font(.body)
                Text(byteString(m.sizeBytes)).font(.caption).foregroundStyle(.secondary)
            }
            CapabilityBadges(capabilities: m.capabilities, loaded: isLoaded(m.descriptor.repoID))
            Spacer()
            loadControl(for: m.descriptor.repoID)
        }
        .padding(.leading, indented ? 16 : 0)
    }

    /// Load / Loading… / Unload (base models) or attachment status (drafters).
    @ViewBuilder
    private func loadControl(for repoID: String) -> some View {
        if isDrafter(repoID) {
            if isAttachedDrafter(repoID) {
                Text("Attached")
                    .font(.caption2).padding(.horizontal, 5).padding(.vertical, 2)
                    .background(Color.green.opacity(0.18), in: Capsule())
                    .foregroundStyle(.green)
                    .help("Attached to the loaded model for MTP speculative decoding.")
            } else {
                Text("drafter").font(.caption2)
                    .padding(.horizontal, 6).padding(.vertical, 3)
                    .foregroundStyle(.secondary)
                    .help("MTP drafter — auto-attaches when its base model is loaded.")
            }
        } else if isLoaded(repoID) {
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
    /// When true, a "Loaded" badge is shown alongside the capability badges (same row).
    var loaded: Bool = false

    var body: some View {
        HStack(spacing: 4) {
            if loaded { badge("Loaded", .green) }
            if capabilities.contains(.vision) { badge("Vision", .purple) }
            if capabilities.contains(.speculative) { badge("MTP", .orange) }
            if capabilities.contains(.tools) { badge("Tools", .blue) }
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
