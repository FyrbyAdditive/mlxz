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

            Section {
                let perf = Bindable(model.perfSettings)
                Picker("KV cache compression", selection: perf.kvBits) {
                    Text("Off (fp16)").tag(0)
                    Text("8-bit").tag(8)
                    Text("4-bit").tag(4)
                }
                Stepper(
                    "Prefix cache slots: \(model.perfSettings.prefixCacheSlots)",
                    value: perf.prefixCacheSlots, in: 0...64)
                Stepper(
                    "Prefix cache RAM cap: \(model.perfSettings.prefixCacheBytesMB) MB",
                    value: perf.prefixCacheBytesMB, in: 0...16384, step: 512)
                Picker("Snapshot block", selection: perf.snapshotBlock) {
                    Text("256 (more reuse)").tag(256)
                    Text("512").tag(512)
                    Text("1024").tag(1024)
                    Text("2048 (less RAM)").tag(2048)
                }
            } header: {
                Text("Performance")
            } footer: {
                Text(
                    "KV compression shrinks the cache (4-bit is lossless for greedy on large models). "
                    + "Prefix cache reuses a shared system prompt across requests; more slots / smaller "
                    + "blocks raise reuse but use a little more memory. Changes apply on the next model load.")
                .font(.caption).foregroundStyle(.secondary)
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
                HStack {
                    if model.serverRunning {
                        Button("Stop Server", role: .destructive) { Task { await model.stopServer() } }
                    } else {
                        Button("Start Server") { Task { await model.startServer() } }
                            .disabled(model.modelState.loadedDescriptor == nil)
                    }
                }
            } footer: {
                if model.modelState.loadedDescriptor == nil {
                    Text("Load a model before starting the server.")
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
        switch model.modelState {
        case .empty: "none loaded"
        case .loading(let d, let f): "loading \(d.displayName) \(f.map { "\(Int($0 * 100))%" } ?? "")"
        case .loaded(let d, let drafter):
            drafter != nil ? "\(d.displayName) + MTP drafter" : d.displayName
        case .failed(_, let msg): "failed: \(msg)"
        }
    }

    private func copyToPasteboard(_ text: String) {
        #if canImport(AppKit)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #endif
    }
}
