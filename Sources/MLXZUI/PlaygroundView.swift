import SwiftUI

/// A minimal chat box that dogfoods the running local server end-to-end.
struct PlaygroundView: View {
    @Bindable var model: AppModel

    @State private var prompt = ""
    @State private var transcript: [Turn] = []
    @State private var streaming = false

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
                    if let last = transcript.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
                }
            }

            Divider()

            HStack {
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
                try await model.playgroundSend(text) { delta in
                    transcript[replyIndex].text += delta
                }
            } catch {
                transcript[replyIndex].text = "⚠️ \(error.localizedDescription)"
            }
        }
    }
}
