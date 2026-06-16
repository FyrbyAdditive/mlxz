import SwiftUI
import MLXZCore

/// Top-level GUI: a sidebar selecting between Models, Server, and Logs.
public struct ContentView: View {
    @State private var model: AppModel
    @State private var selection: Section = .models

    public init(model: AppModel) {
        _model = State(initialValue: model)
    }

    enum Section: String, CaseIterable, Identifiable {
        case models = "Models"
        case server = "Server"
        case logs = "Logs"
        var id: String { rawValue }
        var icon: String {
            switch self {
            case .models: "shippingbox"
            case .server: "network"
            case .logs: "text.alignleft"
            }
        }
    }

    public var body: some View {
        NavigationSplitView {
            List(Section.allCases, selection: $selection) { section in
                Label(section.rawValue, systemImage: section.icon).tag(section)
            }
            .navigationSplitViewColumnWidth(min: 160, ideal: 180)
        } detail: {
            switch selection {
            case .models: ModelLibraryView(model: model)
            case .server: ServerControlView(model: model)
            case .logs: LogsView(model: model)
            }
        }
        .task { await model.observeModelState() }
    }
}
