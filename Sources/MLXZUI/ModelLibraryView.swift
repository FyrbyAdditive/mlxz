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
    @State private var pendingDelete: InstalledModel?

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
                            Text("↓ \(entry.downloads)"
                                + (entry.sizeString.map { "  ·  \($0)" } ?? ""))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        CapabilityBadges(capabilities: entry.capabilities, repoID: entry.id,
                                         loaded: isLoaded(entry.id))
                        Spacer()
                        downloadControl(for: entry.id)
                        loadControl(for: entry.id)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Models")
        .task {
            model.pruneStaleDownloads()   // drop "Retry" for partials the user has since deleted
            installed = model.installedModels()
        }
        // When a download finishes, re-enumerate so the new model shows in the Installed list above
        // immediately (previously it only appeared after switching tabs and back).
        .onChange(of: model.downloads.completedCount) {
            installed = model.installedModels()
        }
        .confirmationDialog(
            "Delete \(pendingDelete?.displayName ?? "model")?",
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
            presenting: pendingDelete
        ) { m in
            Button("Delete \(byteString(m.sizeBytes))", role: .destructive) {
                Task {
                    await model.deleteModel(m)
                    installed = model.installedModels()
                    pendingDelete = nil
                }
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: { m in
            Text("This permanently removes \(m.descriptor.repoID) from disk. You can re-download it later.")
        }
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
            CapabilityBadges(
                capabilities: m.capabilities, repoID: m.descriptor.repoID, modelType: m.modelType,
                loaded: isLoaded(m.descriptor.repoID))
            Spacer()
            loadControl(for: m.descriptor.repoID)
            Button(role: .destructive) { pendingDelete = m } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless).foregroundStyle(.red)
            .help("Delete \(m.displayName) from disk (\(byteString(m.sizeBytes)))")
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

    /// The Download / progress / cancel control for a row, as a dedicated child view (see
    /// `DownloadControl`) so it observes the `DownloadManager` directly and keeps re-rendering even
    /// after this `ModelLibraryView` is torn down and recreated (which happens on every tab switch).
    private func downloadControl(for repoID: String) -> some View {
        DownloadControl(downloads: model.downloads, repoID: repoID)
    }

    private func runSearch() {
        model.pruneStaleDownloads()   // so deleted partials don't show "Retry" in fresh results
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

/// Model type + capability chips: tuning (Instruct/Base/Reasoning/Code), architecture (MoE / Vision /
/// MTP), tools, and quantization — so the list shows at a glance what each model is.
struct CapabilityBadges: View {
    let capabilities: ModelCapabilities
    /// Repo id + optional model_type drive the type labels (tuning, MoE, quant). Empty repoID shows
    /// only capability chips (e.g. for a bare drafter row).
    var repoID: String = ""
    var modelType: String? = nil
    /// When true, a "Loaded" badge is shown alongside the badges (same row).
    var loaded: Bool = false

    var body: some View {
        HStack(spacing: 4) {
            if loaded { badge("Loaded", .green) }
            // Tuning (what the model is for).
            if let t = ModelTypeInfo.tuning(repoID: repoID, modelType: modelType) {
                switch t {
                case .instruct:  badge("Instruct", .blue)
                case .reasoning: badge("Reasoning", .indigo)
                case .base:      badge("Base", .secondaryGray)
                case .code:      badge("Code", .teal)
                }
            }
            // Architecture / modality.
            if ModelTypeInfo.isMoE(repoID: repoID, modelType: modelType) { badge("MoE", .pink) }
            if capabilities.contains(.vision) { badge("Vision", .purple) }
            if capabilities.contains(.speculative) { badge("MTP", .orange) }
            // Quantization (how big / precise).
            if let q = ModelTypeInfo.quantization(repoID: repoID) { badge(q, .brown) }
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

extension Color {
    /// A muted gray that reads as a label color (distinct from `.secondary` foreground usage).
    fileprivate static var secondaryGray: Color { Color.gray }
}

/// Download / progress / cancel control for one repo. A dedicated `View` that observes the
/// `@Observable DownloadManager` directly, so it re-renders on every progress tick and — crucially —
/// keeps working after the parent `ModelLibraryView` is destroyed/recreated (e.g. switching tabs and
/// back), since its observation is rebound to the (stable, shared) manager when it appears.
struct DownloadControl: View {
    let downloads: DownloadManager
    let repoID: String

    var body: some View {
        switch downloads.downloads.first(where: { $0.id == repoID })?.state {
        case .downloading(let fraction, let completed, let total):
            HStack(spacing: 6) {
                ProgressView(value: fraction).frame(width: 60)
                VStack(alignment: .leading, spacing: 0) {
                    Text("\(Int(fraction * 100))%").font(.caption).monospacedDigit()
                    if total > 0 {
                        Text("\(byteString(completed)) / \(byteString(total))")
                            .font(.caption2).foregroundStyle(.secondary).monospacedDigit()
                    }
                }
                Button("✕") { downloads.cancel(repoID) }.buttonStyle(.borderless)
            }
        case .done:
            Label("Downloaded", systemImage: "checkmark.circle").labelStyle(.iconOnly).foregroundStyle(.green)
        case .failed:
            Button("Retry") { downloads.start(repoID) }.foregroundStyle(.red)
        default:
            Button("Download") { downloads.start(repoID) }
        }
    }

    private func byteString(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
