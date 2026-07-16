import SwiftUI

/// A minimal chat box that dogfoods the running local server end-to-end.
struct PlaygroundView: View {
    @Bindable var model: AppModel

    @State private var prompt = ""
    @State private var transcript: [Turn] = []
    @State private var streaming = false
    /// Which loaded model to chat with (routing is strict). Defaults to the first loaded.
    @State private var selectedModel: String?

    struct Turn: Identifiable {
        let id = UUID()
        let role: String
        var text: String
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        ForEach(transcript) { turn in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(turn.role.capitalized).font(.caption).foregroundStyle(.secondary)
                                Text(turn.text).textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .padding(10)
                            .background(turn.role == "user" ? Color.blue.opacity(0.08) : Color.gray.opacity(0.08),
                                        in: RoundedRectangle(cornerRadius: 8))
                            .id(turn.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: transcript.last?.text) {
                    // Autoscroll on each streamed token WITHOUT animation: an animated scroll per
                    // token (≈18+/sec) stacks overlapping animations and makes streaming feel
                    // janky/slow even when generation is fast. Instant scroll stays smooth.
                    if let last = transcript.last { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }

            Divider()

            HStack {
                // Model picker — only shown when more than one model is loaded.
                if model.modelState.loaded.count > 1 {
                    Picker("Model", selection: $selectedModel) {
                        ForEach(model.modelState.loaded, id: \.descriptor.repoID) { m in
                            Text(m.descriptor.displayName).tag(Optional(m.descriptor.repoID))
                        }
                    }
                    .labelsHidden().fixedSize()
                }
                TextField("Message the loaded model…", text: $prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...5)
                    .onSubmit(send)
                Button("Send", action: send)
                    .disabled(streaming || prompt.trimmingCharacters(in: .whitespaces).isEmpty || !model.serverRunning)
            }
            .padding()
        }
        .navigationTitle("Playground")
        .onAppear { if selectedModel == nil { selectedModel = model.modelState.loaded.first?.descriptor.repoID } }
        .overlay(alignment: .top) {
            if !model.serverRunning {
                Text("Start the server to chat.")
                    .font(.caption).padding(6)
                    .background(.yellow.opacity(0.25), in: Capsule())
                    .padding(.top, 4)
            }
        }
    }

    private func send() {
        let text = prompt.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !streaming else { return }
        prompt = ""
        transcript.append(Turn(role: "user", text: text))
        transcript.append(Turn(role: "assistant", text: ""))
        let replyIndex = transcript.count - 1
        streaming = true

        Task {
            defer { streaming = false }
            do {
                try await model.playgroundSend(text, model: selectedModel) { delta in
                    transcript[replyIndex].text += delta
                }
            } catch {
                transcript[replyIndex].text = "⚠️ \(error.localizedDescription)"
            }
        }
    }
}
