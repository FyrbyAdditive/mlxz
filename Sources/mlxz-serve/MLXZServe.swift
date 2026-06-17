import Foundation
import ArgumentParser
import Logging
import MLXZCore
import MLXZInference
import MLXZServer
import MLXZHub

/// Headless OpenAI-compatible server. The composition root: wires the real MLX loader into
/// the ModelManager and starts the Hummingbird server. Used for curl/SDK smoke tests and as
/// the engine behind the GUI.
@main
struct MLXZServe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlxz-serve",
        abstract: "Serve a local MLX model over an OpenAI-compatible API."
    )

    @Option(name: .long, help: "HuggingFace repo id, e.g. mlx-community/Qwen3.6-4B-4bit")
    var model: String

    @Option(name: .long, help: "Bind host (use 0.0.0.0 for LAN).")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Bind port.")
    var port: Int = 8080

    @Option(name: .long, help: "Optional API key required in the Authorization: Bearer header.")
    var apiKey: String?

    @Option(name: .long, help: "Quantize the KV cache to N bits (e.g. 8) to cut memory ~2-3.5x. Best on large models; degrades small models — leave off unless memory-bound.")
    var kvBits: Int?

    @Option(name: .long, help: "Cap the KV cache to N tokens (rotating cache); bounds memory on long chats. Disables prefix-cache reuse.")
    var maxKvSize: Int?

    @Flag(name: .long, inversion: .prefixedNo, help: "Reuse the KV cache for shared prompt prefixes across requests.")
    var prefixCache: Bool = true

    @Flag(name: .long, inversion: .prefixedNo, help: "Use native MTP self-speculative decoding for MTP-capable models.")
    var mtp: Bool = true

    @Option(name: .long, help: "Attach a standalone MTP drafter checkpoint (repo id) to the model for draft-model speculative decoding.")
    var mtpDraft: String?

    @Option(name: .long, help: "Bound MLX's GPU buffer cache to N MB (default 512). Prevents MLX hoarding multi-GB of scratch buffers next to a large model. 0 disables the cache.")
    var gpuCacheMb: Int = 512

    @Flag(name: .long, help: "Print the VS Code Copilot model-config snippet and exit.")
    var printCopilotConfig: Bool = false

    func run() async throws {
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardOutput(label: label)
            handler.logLevel = .info
            return handler
        }
        let logger = Logger(label: "mlxz")

        let descriptor = ModelDescriptor(repoID: model)

        if printCopilotConfig {
            let caps = ModelCapabilityDetector.detect(repoID: model)
            print(CopilotConfig.setupHint(host: host, port: port))
            print()
            print(CopilotConfig.modelEntry(repoID: model, host: host, port: port, capabilities: caps))
            return
        }

        let perf = EnginePerfOptions(
            kvBits: kvBits,
            maxKVSize: maxKvSize,
            prefixCache: prefixCache,
            useMTP: mtp,
            gpuCacheLimitMB: gpuCacheMb
        )
        let manager = ModelManager(
            loader: MLXModelLoader(perf: perf, draftModelID: mtpDraft), logger: logger)

        logger.info("loading model (first run downloads from HuggingFace)…", metadata: ["model": .string(model)])
        try await manager.load(descriptor)
        logger.info("model ready", metadata: ["model": .string(model)])

        let localStore = LocalModelStore()
        let embeddingManager = EmbeddingManager(loader: MLXEmbeddingLoader())
        let server = InferenceServer(
            manager: manager,
            logger: logger,
            logSink: { line in logger.info("\(line)") },
            extraModelIDs: { localStore.installedModels().map(\.descriptor.repoID) },
            embeddingManager: embeddingManager,
            metricsSink: { usage in
                if let tps = usage.tokensPerSecond {
                    logger.info("generated \(usage.completionTokens) tokens at \(String(format: "%.1f", tps)) tok/s")
                }
            }
        )
        try await server.start(ServerConfig(host: host, port: port, apiKey: apiKey))

        // Print a ready banner with the Copilot snippet.
        let caps = await manager.currentEngine()?.capabilities ?? [.chat, .tools]
        print("\n\(CopilotConfig.setupHint(host: host, port: port))\n")
        print(CopilotConfig.modelEntry(repoID: model, host: host, port: port, capabilities: caps))
        print("\nServing. Press Ctrl-C to stop.\n")

        // Run until cancelled (Ctrl-C).
        try await Task.sleep(for: .seconds(60 * 60 * 24 * 365))
    }
}
