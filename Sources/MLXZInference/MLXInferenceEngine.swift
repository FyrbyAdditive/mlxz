import Foundation
import MLXZCore
import MLXLMCommon

/// An `InferenceEngine` backed by an mlx-swift-lm `ModelContainer`. This is the only type
/// that touches MLX. It maps our wire-independent `GenerationRequest` onto the package's
/// stateless `generate(input:parameters:)` path and translates `Generation` events back.
///
/// Speculative decoding (`request.speculative`): the seam is present, but mlx-swift-lm 3.31.3's
/// `ModelContainer.generate` does not expose speculative decoding — `SpeculativeTokenIterator`
/// requires a *separate* draft model and there is no native single-model MTP support yet. So a
/// speculative request is served with standard decoding (correct output, just no speedup) and a
/// note is logged. Capability `.speculative` is still advertised so the UI/Copilot can surface it;
/// when upstream lands MTP/draft support, only this file changes.
public struct MLXInferenceEngine: InferenceEngine {
    public let descriptor: ModelDescriptor
    public let capabilities: ModelCapabilities

    private let container: ModelContainer

    public init(descriptor: ModelDescriptor, capabilities: ModelCapabilities, container: ModelContainer) {
        self.descriptor = descriptor
        self.capabilities = capabilities
        self.container = container
    }

    public func generate(_ request: GenerationRequest) async throws -> AsyncThrowingStream<GenerationEvent, Error> {
        let parameters = Self.mapParameters(request.sampling, maxTokens: request.maxTokens)
        let modelID = descriptor.repoID
        let container = self.container

        // Capture only Sendable values (`request`, `container`, `parameters`). The non-Sendable
        // `UserInput`/`LMInput` are built and consumed entirely inside the task.
        return AsyncThrowingStream { continuation in
            let task = Task {
                continuation.yield(.started(.init(modelID: modelID)))
                // Fallback parser for models whose tool calls arrive as raw <tool_call> text
                // rather than as native `.toolCall` events.
                var fallback = ToolCallParser()
                var sawNativeToolCall = false
                do {
                    let chat = request.messages.map(Self.mapMessage)
                    let tools = request.tools.map { $0.map(Self.mapTool) }
                    let userInput = UserInput(chat: chat, tools: tools)
                    let lmInput = try await container.prepare(input: userInput)
                    let stream = try await container.generate(input: lmInput, parameters: parameters)
                    for await item in stream {
                        if Task.isCancelled { break }
                        switch item {
                        case .chunk(let text):
                            let parsed = fallback.consume(text)
                            if !parsed.visibleText.isEmpty {
                                continuation.yield(.textDelta(parsed.visibleText))
                            }
                            for call in parsed.toolCalls {
                                continuation.yield(.toolCall(call))
                            }

                        case .toolCall(let nativeCall):
                            sawNativeToolCall = true
                            continuation.yield(.toolCall(Self.mapNativeToolCall(nativeCall)))

                        case .info(let info):
                            let tail = fallback.finish()
                            if !tail.visibleText.isEmpty {
                                continuation.yield(.textDelta(tail.visibleText))
                            }
                            for call in tail.toolCalls {
                                continuation.yield(.toolCall(call))
                            }
                            let usage = TokenUsage(
                                promptTokens: info.promptTokenCount,
                                completionTokens: info.generationTokenCount,
                                tokensPerSecond: info.tokensPerSecond
                            )
                            let reason = Self.mapStopReason(info.stopReason, sawToolCall: sawNativeToolCall)
                            continuation.yield(.completed(.init(finishReason: reason, usage: usage)))

                        @unknown default:
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
