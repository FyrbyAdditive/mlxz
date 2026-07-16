import SwiftUI
import MLXZCore
import MLXZHub

/// The "Discover" pane: live-searching the HuggingFace MLX catalog with facet filters, a
/// sort menu, and an author-scope toggle, rendering rich result cards with a RAM-fit hint.
/// Handles the loading / empty / error states the old Form silently dropped.
struct DiscoverView: View {
    @Bindable var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            controls
            Divider()
            content
        }
        .task { await model.loadInitialDiscoverIfNeeded() }
    }

    // MARK: - Controls (search + facets + sort + scope)

    private var controls: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search models…", text: $model.searchQuery)
                    .textFieldStyle(.plain)
                    .onChange(of: model.searchQuery) { model.scheduleSearch() }
                    .onSubmit { runNow() }
                if model.isSearching { ProgressView().controlSize(.small) }
            }
            .padding(8)
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))

            HStack {
                // Facet chips
                ForEach(ModelFacet.allCases) { f in
                    facetChip(f)
                }
                Spacer()
                // Sort
                Menu {
                    Picker("Sort", selection: $model.sortKey) {
                        Text("Popular").tag(CatalogSort.popular)
                        Text("Most liked").tag(CatalogSort.liked)
                        Text("Recently updated").tag(CatalogSort.recentlyUpdated)
                        Text("Smallest").tag(CatalogSort.sizeSmallest)
                        Text("Largest").tag(CatalogSort.sizeLargest)
                    }
                } label: {
                    Label(sortLabel, systemImage: "arrow.up.arrow.down")
                }
                .menuStyle(.borderlessButton).fixedSize()
                .onChange(of: model.sortKey) { runNow() }

                // Author scope
                Toggle(isOn: Binding(
                    get: { !model.mlxCommunityOnly },
                    set: { model.mlxCommunityOnly = !$0 }
                )) {
                    Text("All authors").font(.caption)
                }
                .toggleStyle(.switch).controlSize(.mini)
                .onChange(of: model.mlxCommunityOnly) { runNow() }
                .help("Search all MLX authors, or just the curated mlx-community org.")
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
    }

    private func facetChip(_ f: ModelFacet) -> some View {
        let selected = model.facet == f
        return Button {
            model.facet = f
        } label: {
            Text(f.rawValue).font(.caption)
                .padding(.horizontal, 9).padding(.vertical, 4)
                .background(selected ? Color.accentColor : Color.secondary.opacity(0.15), in: Capsule())
                .foregroundStyle(selected ? Color.white : Color.primary)
        }
        .buttonStyle(.plain)
    }

    private var sortLabel: String {
        switch model.sortKey {
        case .popular: "Popular"
        case .liked: "Most liked"
        case .recentlyUpdated: "Recently updated"
        case .sizeSmallest: "Smallest"
        case .sizeLargest: "Largest"
        }
    }

    // MARK: - Content (results / loading / empty / error)

    @ViewBuilder
    private var content: some View {
        if let error = model.searchError {
            errorState(error)
        } else if visibleResults.isEmpty && !model.isSearching {
            emptyState
        } else {
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(visibleResults) { entry in
                        ModelCard(
                            model: model,
                            repoID: entry.id,
                            title: entry.displayName,
                            subtitle: entry.author,
                            capabilities: entry.capabilities,
                            modelType: nil,
                            metadata: metadata(for: entry),
                            ramFit: RAMFit.of(sizeBytes: entry.sizeBytes.map(Int64.init))
                        )
                    }
                }
                .padding(16)
            }
        }
    }

    private func errorState(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "wifi.slash").font(.system(size: 36)).foregroundStyle(.secondary)
            Text(message).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
            Button("Retry") { runNow() }.buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).padding(40)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "magnifyingglass").font(.system(size: 34)).foregroundStyle(.secondary)
            Text(model.searchQuery.isEmpty ? "No models found" : "No models match “\(model.searchQuery)”")
                .font(.callout).foregroundStyle(.secondary)
            if model.facet != .all {
                Button("Clear filter") { model.facet = .all }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).padding(40)
    }

    /// Drafter-hidden + facet-filtered results (server already sorted/scoped them).
    private var visibleResults: [CatalogEntry] {
        DiscoverFilter.apply(model.discoverResults, facet: model.facet)
    }

    private func metadata(for e: CatalogEntry) -> [String] {
        var parts: [String] = []
        if let cls = e.sizeClass { parts.append(cls) }
        if let size = ByteFormat.string(e.sizeBytes) { parts.append(size) }
        parts.append("\(CountFormat.compact(e.downloads)) ↓")
        if e.likes > 0 { parts.append("♥ \(CountFormat.compact(e.likes))") }
        if let rel = relativeDate(e.lastModified) { parts.append(rel) }
        return parts
    }

    private func relativeDate(_ iso: String?) -> String? {
        guard let iso, let date = ISO8601DateFormatter().date(from: iso) else { return nil }
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .abbreviated
        return f.localizedString(for: date, relativeTo: Date())
    }

    private func runNow() {
        Task {
            await model.runSearch(
                query: model.searchQuery,
                author: model.mlxCommunityOnly ? "mlx-community" : nil,
                sort: model.sortKey)
        }
    }
}
