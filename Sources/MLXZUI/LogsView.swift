import SwiftUI

/// A live scrolling view of the server/model log lines.
struct LogsView: View {
    @Bindable var model: AppModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(model.logStore.lines) { line in
                        Text(line.text)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(line.id)
                    }
                }
                .padding(8)
            }
            .onChange(of: model.logStore.lines.count) {
                if let last = model.logStore.lines.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
        .navigationTitle("Logs")
        .toolbar {
            Button("Clear") { model.logStore.clear() }
        }
    }
}
