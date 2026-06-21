import Testing
@testable import MLXZCore

@Suite struct ModelTypeInfoTests {
    @Test func tuningInstructVsBase() {
        // The user's case: gemma-4-31b-it-8bit is Instruct; gemma-4-31b-8bit has no instruct marker.
        #expect(ModelTypeInfo.tuning(repoID: "mlx-community/gemma-4-31b-it-8bit") == .instruct)
        #expect(ModelTypeInfo.tuning(repoID: "mlx-community/Qwen3.6-27B-Instruct-4bit") == .instruct)
        #expect(ModelTypeInfo.tuning(repoID: "meta-llama/Llama-3.1-8B-chat") == .instruct)
        // Explicit base.
        #expect(ModelTypeInfo.tuning(repoID: "mlx-community/gemma-4-31b-base-8bit") == .base)
        // No clear marker → nil (don't guess "Base" for a bare id).
        #expect(ModelTypeInfo.tuning(repoID: "mlx-community/gemma-4-31b-8bit") == nil)
    }

    @Test func tuningReasoningAndCode() {
        #expect(ModelTypeInfo.tuning(repoID: "mlx-community/DeepSeek-R1-Distill-8B") == .reasoning)
        #expect(ModelTypeInfo.tuning(repoID: "mlx-community/QwQ-32B-4bit") == .reasoning)
        #expect(ModelTypeInfo.tuning(repoID: "mlx-community/Qwen2.5-Coder-7B-Instruct") == .code)
    }

    @Test func moeDetection() {
        #expect(ModelTypeInfo.isMoE(repoID: "mlx-community/Qwen3.6-35B-A3B-4bit"))
        #expect(ModelTypeInfo.isMoE(repoID: "x", modelType: "qwen3_moe"))
        #expect(ModelTypeInfo.isMoE(repoID: "mlx-community/Mixtral-8x7B-Instruct"))
        #expect(!ModelTypeInfo.isMoE(repoID: "mlx-community/gemma-4-31b-it-8bit"))
    }

    @Test func quantizationLabels() {
        #expect(ModelTypeInfo.quantization(repoID: "mlx-community/gemma-4-31b-it-8bit") == "8-bit")
        #expect(ModelTypeInfo.quantization(repoID: "mlx-community/Qwen3.6-27B-4bit") == "4-bit")
        #expect(ModelTypeInfo.quantization(repoID: "mlx-community/gemma-4-31b-it-bf16") == "BF16")
        #expect(ModelTypeInfo.quantization(repoID: "mlx-community/gpt-oss-20b-MXFP4-Q8") == "MXFP4")
        #expect(ModelTypeInfo.quantization(repoID: "some/model-no-quant") == nil)
    }
}
