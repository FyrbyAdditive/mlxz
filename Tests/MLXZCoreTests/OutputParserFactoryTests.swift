import Testing
@testable import MLXZCore

@Suite struct OutputParserFactoryTests {
    @Test func detectsHarmonyForGptOss() {
        #expect(OutputParserFactory.detectFormat(repoID: "mlx-community/gpt-oss-20b-MXFP4-Q8") == .harmony)
        #expect(OutputParserFactory.detectFormat(repoID: "openai/gpt-oss-120b") == .harmony)
        #expect(OutputParserFactory.detectFormat(repoID: "x", modelType: "gpt_oss") == .harmony)
    }

    @Test func detectsMistral() {
        #expect(OutputParserFactory.detectFormat(repoID: "x", modelType: "mistral3") == .mistral)
        #expect(OutputParserFactory.detectFormat(repoID: "mlx-community/Ministral-8B-Instruct") == .mistral)
    }

    @Test func detectsGLM4() {
        #expect(OutputParserFactory.detectFormat(repoID: "x", modelType: "glm4_moe") == .glm4)
        #expect(OutputParserFactory.detectFormat(repoID: "mlx-community/GLM-4-9B") == .glm4)
    }

    @Test func detectsLlama3() {
        #expect(OutputParserFactory.detectFormat(repoID: "mlx-community/Meta-Llama-3-8B") == .llama3)
        #expect(OutputParserFactory.detectFormat(repoID: "x", modelType: "llama4") == .llama3)
    }

    @Test func detectsGemma() {
        #expect(OutputParserFactory.detectFormat(repoID: "mlx-community/gemma-4-31b-it-4bit") == .gemma)
        #expect(OutputParserFactory.detectFormat(repoID: "google/gemma-3-4b-it") == .gemma)
        #expect(OutputParserFactory.detectFormat(repoID: "x", modelType: "gemma") == .gemma)
    }

    @Test func defaultsToQwenHermes() {
        #expect(OutputParserFactory.detectFormat(repoID: "mlx-community/Qwen3.6-27B-4bit") == .qwenHermes)
        #expect(OutputParserFactory.detectFormat(repoID: "mlx-community/Hermes-3") == .qwenHermes)
        #expect(OutputParserFactory.detectFormat(repoID: "") == .qwenHermes)
        #expect(OutputParserFactory.detectFormat(repoID: "some/unknown-model") == .qwenHermes)
    }

    @Test func makeReturnsExpectedTypes() {
        #expect(OutputParserFactory.make(format: .qwenHermes, startInsideThink: false) is QwenOutputParser)
        #expect(OutputParserFactory.make(format: .harmony, startInsideThink: false) is HarmonyOutputParser)
        #expect(OutputParserFactory.make(format: .mistral, startInsideThink: false) is MistralOutputParser)
        #expect(OutputParserFactory.make(format: .llama3, startInsideThink: false) is Llama3OutputParser)
        #expect(OutputParserFactory.make(format: .glm4, startInsideThink: false) is GLM4OutputParser)
        #expect(OutputParserFactory.make(format: .gemma, startInsideThink: false) is GemmaOutputParser)
    }

    @Test func formatHooks() {
        // Only Qwen pre-opens a <think> block.
        #expect(OutputFormat.qwenHermes.prefersPreOpenedThink)
        for f in [OutputFormat.harmony, .mistral, .llama3, .glm4, .gemma] {
            #expect(!f.prefersPreOpenedThink)
        }
        // enable_thinking kwarg: Qwen and Gemma understand it; others must not be sent it.
        #expect(OutputFormat.qwenHermes.supportsEnableThinkingKwarg)
        #expect(OutputFormat.gemma.supportsEnableThinkingKwarg)
        for f in [OutputFormat.harmony, .mistral, .llama3, .glm4] {
            #expect(!f.supportsEnableThinkingKwarg)
        }
        // Only Gemma surfaces a thought channel as reasoning (when thinking is on).
        #expect(OutputFormat.gemma.thoughtChannelIsReasoningWhenThinking)
        #expect(!OutputFormat.qwenHermes.thoughtChannelIsReasoningWhenThinking)
    }
}
