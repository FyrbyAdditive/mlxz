import SwiftUI
import MLXZCore

/// Performance & memory tuning (KV-cache compression, prefix cache, reasoning budget, MTP drafter,
/// image resolution). Its own sidebar area; settings apply on the next model load.
struct PerformanceView: View {
    @Bindable var model: AppModel

    var body: some View {
        Form {
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
                Picker("Reasoning budget", selection: perf.reasoningTokenBudget) {
                    Text("Uncapped").tag(0)
                    Text("256 (low)").tag(256)
                    Text("1024 (medium)").tag(1024)
                    Text("2048").tag(2048)
                    Text("4096 (high)").tag(4096)
                }
                Toggle("Use MTP drafter (self-speculative)", isOn: perf.useMTPDrafter)
                Picker("Max image resolution", selection: perf.maxImagePixels) {
                    Text("1 MP (low mem)").tag(1_048_576)
                    Text("4 MP (recommended)").tag(4_194_304)
                    Text("8 MP (high detail)").tag(8_388_608)
                }
            } footer: {
                Text(
                    "KV compression shrinks the cache (4-bit is lossless for greedy on large models). "
                    + "Prefix cache reuses a shared system prompt across requests; more slots / smaller "
                    + "blocks raise reuse but use a little more memory. Reasoning budget caps the model's "
                    + "<think> block — it force-closes reasoning after N tokens and makes the model answer "
                    + "(a request's reasoning_effort overrides this). MTP drafter auto-attaches a matching "
                    + "installed drafter for faster decoding; turn off to load the base model alone. "
                    + "Changes apply on the next model load.")
                .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Performance")
    }
}
