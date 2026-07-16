import SwiftUI
import MLXZCore
import MLXZHub

/// A full-width card row used by both panes. It renders name + metadata + capability
/// badges, and a single primary action derived from the unified `ModelStatus` so an
/// already-installed search hit shows "Load", never "Download". `metadata`/`trailingSize`
/// let the two call sites feed the fields they have (Discover has downloads/likes/RAM-fit;
/// My Models has on-disk size).
struct ModelCard: View {
    @Bindable var model: AppModel
    let repoID: String
    let title: String
    let subtitle: String?
    let capabilities: ModelCapabilities
    let modelType: String?
    /// Extra metadata chips (downloads/likes/updated/size-class) shown under the title.
    let metadata: [String]
    /// RAM-fit advisory (Discover only); `.unknown` renders nothing.
    var ramFit: RAMFit = .unknown
    /// Indent (a drafter nested under its base model).
    var indented: Bool = false
    /// Called for the delete affordance (My Models only); nil hides it.
    var onDelete: (() -> Void)? = nil

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            if indented {
                Image(systemName: "arrow.turn.down.right")
                    .font(.caption).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(title).font(.body.weight(.medium)).lineLimit(1)
                    CapabilityBadges(capabilities: capabilities, repoID: repoID, modelType: modelType)
                }
                if let subtitle {
                    Text(subtitle).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                if !metadata.isEmpty || ramFit != .unknown {
                    HStack(spacing: 10) {
                        ForEach(metadata, id: \.self) { m in
                            Text(m).font(.caption2).foregroundStyle(.secondary)
                        }
                        RAMFitBadge(fit: ramFit)
                    }
                }
            }
            Spacer(minLength: 8)
            primaryAction
            if let onDelete {
                Button(role: .destructive, action: onDelete) {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless).foregroundStyle(.red)
                .help("Delete from disk")
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .padding(.leading, indented ? 20 : 0)
    }

    @ViewBuilder
    private var primaryAction: some View {
        switch model.status(of: repoID) {
        case .loaded:
            HStack(spacing: 6) {
                Label("Loaded", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                    .labelStyle(.titleAndIcon).font(.caption)
                Button("Unload", role: .destructive) { Task { await model.unload(repoID) } }
            }
        case .loading:
            HStack(spacing: 4) {
                ProgressView().controlSize(.small)
                Text("Loading…").font(.caption).foregroundStyle(.secondary)
            }
        case .drafter(let attached):
            Text(attached ? "Attached" : "drafter")
                .font(.caption2).padding(.horizontal, 6).padding(.vertical, 3)
                .background(attached ? Color.green.opacity(0.18) : Color.clear, in: Capsule())
                .foregroundStyle(attached ? .green : .secondary)
                .help(attached
                      ? "Attached to the loaded model for speculative decoding."
                      : "Speculative companion — auto-attaches when its base model loads.")
        case .downloading(let fraction):
            HStack(spacing: 6) {
                ProgressView(value: fraction).frame(width: 70)
                Text("\(Int(fraction * 100))%").font(.caption).monospacedDigit()
                Button {
                    model.cancelDownload(repoID)
                } label: { Image(systemName: "xmark.circle.fill") }
                    .buttonStyle(.borderless).foregroundStyle(.secondary)
            }
        case .installed:
            Button("Load") { Task { await model.load(ModelDescriptor(repoID: repoID)) } }
                .buttonStyle(.borderedProminent).controlSize(.small)
        case .downloadFailed:
            Button("Retry") { model.startDownload(repoID) }.foregroundStyle(.red)
        case .remote:
            Button("Get") { model.startDownload(repoID) }
                .buttonStyle(.bordered).controlSize(.small)
        }
    }
}

/// A small colored advisory that a model comfortably fits / is tight / won't fit in RAM.
private struct RAMFitBadge: View {
    let fit: RAMFit
    var body: some View {
        switch fit {
        case .unknown: EmptyView()
        case .fits:
            Label("fits", systemImage: "checkmark").font(.caption2).foregroundStyle(.green)
        case .tight:
            Label("tight", systemImage: "exclamationmark.triangle").font(.caption2).foregroundStyle(.orange)
        case .exceeds:
            Label("exceeds RAM", systemImage: "exclamationmark.octagon").font(.caption2).foregroundStyle(.red)
        }
    }
}

/// Model type + capability chips: tuning (Instruct/Base/Reasoning/Code), architecture
/// (MoE / Vision / MTP), and quantization — so a card shows at a glance what a model is.
struct CapabilityBadges: View {
    let capabilities: ModelCapabilities
    var repoID: String = ""
    var modelType: String? = nil

    var body: some View {
        HStack(spacing: 4) {
            if let t = ModelTypeInfo.tuning(repoID: repoID, modelType: modelType) {
                switch t {
                case .instruct:  badge("Instruct", .blue)
                case .reasoning: badge("Reasoning", .indigo)
                case .base:      badge("Base", .gray)
                case .code:      badge("Code", .teal)
                }
            }
            if ModelTypeInfo.isMoE(repoID: repoID, modelType: modelType) { badge("MoE", .pink) }
            if capabilities.contains(.vision) { badge("Vision", .purple) }
            if capabilities.contains(.speculative) { badge("MTP", .orange) }
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
