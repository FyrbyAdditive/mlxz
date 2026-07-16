import SwiftUI
import MLXZCore
import MLXZHub

/// The Models screen: a pinned "currently loaded" header, a My Models / Discover switch,
/// and the matching pane. Replaces the old three-section scrolling Form with a
/// card/browser-style layout. Search/installed state lives in `AppModel` so it survives
/// tab switches.
struct ModelLibraryView: View {
    @Bindable var model: AppModel

    enum Pane: String, CaseIterable, Identifiable {
        case myModels = "My Models"
        case discover = "Discover"
        var id: String { rawValue }
    }
    @State private var pane: Pane = .myModels
    @State private var showAddByID = false

    var body: some View {
        VStack(spacing: 0) {
            LoadedModelHeader(model: model)
            Divider()

            Picker("", selection: $pane) {
                ForEach(Pane.allCases) { p in Text(p.rawValue).tag(p) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 16).padding(.vertical, 10)

            switch pane {
            case .myModels:
                MyModelsView(model: model, switchToDiscover: { pane = .discover })
            case .discover:
                DiscoverView(model: model)
            }
        }
        .navigationTitle("Models")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showAddByID = true
                } label: { Label("Add by ID", systemImage: "plus") }
                .help("Load or download a model by its HuggingFace repo id")
            }
        }
        .sheet(isPresented: $showAddByID) { AddByIDSheet(model: model) }
        .task {
            model.refreshInstalled()
            model.pruneStaleDownloads()
        }
        .onChange(of: model.downloads.completedCount) { model.refreshInstalled() }
        // Land first-run users (empty library) straight in Discover.
        .onAppear {
            if model.installed.isEmpty { pane = .discover }
        }
    }
}

/// The always-visible header showing the currently-loaded model (or a slim empty strip).
private struct LoadedModelHeader: View {
    @Bindable var model: AppModel

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: loaded == nil ? "shippingbox" : "cpu")
                .font(.title2)
                .foregroundStyle(loaded == nil ? Color.secondary : Color.green)
                .frame(width: 28)
            VStack(alignment: .leading, spacing: 3) {
                if let d = loaded {
                    Text(d.displayName).font(.headline)
                    HStack(spacing: 6) {
                        if let spec = model.speculationStatus {
                            Label(spec.hasPrefix("DSpark") ? "DSpark" : "MTP", systemImage: "bolt.fill")
                                .font(.caption2)
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(Color.green.opacity(0.18), in: Capsule())
                                .foregroundStyle(.green)
                        } else {
                            Text("Loaded").font(.caption).foregroundStyle(.secondary)
                        }
                        Text(d.repoID).font(.caption).foregroundStyle(.secondary)
                    }
                } else {
                    Text("No model loaded").font(.headline).foregroundStyle(.secondary)
                    Text("Load one from My Models, or find a new one in Discover.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer()
            if loaded != nil {
                Button("Unload", role: .destructive) { Task { await model.unload() } }
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 12)
        .background(.regularMaterial)
    }

    private var loaded: ModelDescriptor? { model.modelState.loadedDescriptor }
}

/// A sheet for the power-user "load/download by repo id" escape hatch (relocated from the
/// old top-of-form field). Accepts any repo id, including non-mlx-community orgs.
private struct AddByIDSheet: View {
    @Bindable var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var repoID = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add a model by repo id").font(.headline)
            Text("Paste a HuggingFace repo id (e.g. mlx-community/Qwen3-8B-4bit). "
                 + "Loading downloads it first if it isn't installed.")
                .font(.caption).foregroundStyle(.secondary)
            TextField("org/model-name", text: $repoID)
                .textFieldStyle(.roundedBorder)
                .onSubmit(load)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Download") {
                    model.startDownload(trimmed); dismiss()
                }.disabled(trimmed.isEmpty)
                Button("Load") { load() }
                    .keyboardShortcut(.defaultAction).disabled(trimmed.isEmpty)
            }
        }
        .padding(20).frame(width: 460)
    }

    private var trimmed: String { repoID.trimmingCharacters(in: .whitespaces) }
    private func load() {
        guard !trimmed.isEmpty else { return }
        Task { await model.load(ModelDescriptor(repoID: trimmed)) }
        dismiss()
    }
}
