import SwiftUI
import MLXZCore

/// Start/stop the server, configure binding, and copy the VS Code Copilot snippet.
struct ServerControlView: View {
    @Bindable var model: AppModel

    var body: some View {
        Form {
            Section("Status") {
                LabeledContent("Model") {
                    Text(modelStatusText).foregroundStyle(.secondary)
                }
                LabeledContent("Server") {
                    HStack {
                        Circle().fill(model.serverRunning ? .green : .secondary).frame(width: 8, height: 8)
                        Text(model.serverRunning ? "Running on \(model.bindLAN ? "0.0.0.0" : model.host):\(String(model.port))" : "Stopped")
                            .foregroundStyle(.secondary)
                    }
                }
                if model.serverRunning {
                    LabeledContent("Requests served") {
                        Text("\(model.requestsServed)").foregroundStyle(.secondary)
                    }
                    if let tps = model.lastTokensPerSecond {
                        LabeledContent("Last speed") {
                            Text("\(tps, format: .number.precision(.fractionLength(1))) tok/s").foregroundStyle(.secondary)
                        }
                    }
                }
            }

            Section("Binding") {
                TextField("Host", text: $model.host).disabled(model.bindLAN)
                TextField("Port", value: $model.port, format: .number.grouping(.never))
                Toggle("Allow LAN access (bind 0.0.0.0)", isOn: $model.bindLAN)
                SecureField("API key (optional)", text: $model.apiKey)
                if model.bindLAN && model.apiKey.isEmpty {
                    Label("Binding to the LAN without an API key exposes the server to your network.",
                          systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                }
            }

            Section {
                Toggle("Start server automatically on launch", isOn: $model.autoStartServer)
            } footer: {
                Text("The server starts when the app opens, using the binding above. "
                    + "It can run without a model loaded — requests return a \"no model loaded\" "
                    + "error until you load one.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section {
                HStack {
                    if model.serverRunning {
                        Button("Stop Server", role: .destructive) { Task { await model.stopServer() } }
                    } else {
                        Button("Start Server") { Task { await model.startServer() } }
                    }
                }
            } footer: {
                if model.modelState.loaded.isEmpty {
                    Text("No model is loaded — the server will respond with a \"no model loaded\" "
                        + "error until you load one from the Models tab.")
                }
            }

            if let snippet = model.copilotConfigSnippet() {
                Section("VS Code Copilot (BYOK)") {
                    Text("Add a Custom (OpenAI-compatible) endpoint, then paste this into chatLanguageModels.json:")
                        .font(.caption).foregroundStyle(.secondary)
                    Text(snippet)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 6))
                    Button("Copy snippet") {
                        copyToPasteboard(snippet)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Server")
    }

    private var modelStatusText: String {
        let s = model.modelState
        if !s.loaded.isEmpty {
            if s.loaded.count == 1 {
                let m = s.loaded[0]
                return m.drafterID != nil ? "\(m.descriptor.displayName) + MTP drafter" : m.descriptor.displayName
            }
            return "\(s.loaded.count) models loaded"
        }
        if let l = s.loading.first {
            return "loading \(l.descriptor.displayName) \(l.fraction.map { "\(Int($0 * 100))%" } ?? "")"
        }
        if let f = s.failed.first { return "failed: \(f.message)" }
        return "none loaded"
    }

    private func copyToPasteboard(_ text: String) {
        #if canImport(AppKit)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #endif
    }
}
