import SwiftUI
import MLXZCore
import MLXZHub

/// The "My Models" pane: installed models as cards, drafters nested under their base
/// model, with load/unload/delete. Empty state nudges the user to Discover.
struct MyModelsView: View {
    @Bindable var model: AppModel
    let switchToDiscover: () -> Void

    @State private var pendingDelete: InstalledModel?

    var body: some View {
        Group {
            if model.installed.isEmpty {
                emptyState
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(baseModels) { m in
                            card(for: m, indented: false)
                            if let drafter = installedDrafter(forBase: m.descriptor.repoID) {
                                card(for: drafter, indented: true)
                            }
                        }
                        ForEach(orphanDrafters) { d in card(for: d, indented: false) }
                    }
                    .padding(16)
                }
            }
        }
        .confirmationDialog(
            "Delete \(pendingDelete?.displayName ?? "model")?",
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
            presenting: pendingDelete
        ) { m in
            Button("Delete \(ByteFormat.string(m.sizeBytes))", role: .destructive) {
                Task {
                    await model.deleteModel(m)
                    model.refreshInstalled()
                    pendingDelete = nil
                }
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: { m in
            Text("This permanently removes \(m.descriptor.repoID) from disk. You can re-download it later.")
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "shippingbox").font(.system(size: 40)).foregroundStyle(.secondary)
            Text("No models yet").font(.title3.weight(.medium))
            Text("Find and download an open model to run on your Mac.")
                .font(.callout).foregroundStyle(.secondary)
            Button("Browse Discover") { switchToDiscover() }
                .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }

    private func card(for m: InstalledModel, indented: Bool) -> some View {
        ModelCard(
            model: model,
            repoID: m.descriptor.repoID,
            title: m.displayName,
            subtitle: m.descriptor.repoID,
            capabilities: m.capabilities,
            modelType: m.modelType,
            metadata: [ByteFormat.string(m.sizeBytes)],
            indented: indented,
            onDelete: DrafterPairing.isDrafter(m.descriptor.repoID) ? nil : { pendingDelete = m }
        )
    }

    // MARK: - Installed grouping (base models with their drafter nested beneath)

    private var baseModels: [InstalledModel] {
        model.installed.filter { !DrafterPairing.isDrafter($0.descriptor.repoID) }
    }

    private func installedDrafter(forBase baseRepoID: String) -> InstalledModel? {
        let expected = DrafterPairing.drafterRepoID(forBase: baseRepoID)
        return model.installed.first { $0.descriptor.repoID == expected }
    }

    private var orphanDrafters: [InstalledModel] {
        let baseIDs = Set(baseModels.map(\.descriptor.repoID))
        return model.installed.filter {
            DrafterPairing.isDrafter($0.descriptor.repoID)
                && DrafterPairing.baseRepoID(forDrafter: $0.descriptor.repoID).map {
                    !baseIDs.contains($0)
                } ?? true
        }
    }
}
