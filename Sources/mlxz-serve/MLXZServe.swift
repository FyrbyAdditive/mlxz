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

        let manager = ModelManager(loader: MLXModelLoader(), logger: logger)

        logger.info("loading model (first run downloads from HuggingFace)…", metadata: ["model": .string(model)])
        try await manager.load(descriptor)
        logger.info("model ready", metadata: ["model": .string(model)])

        let server = InferenceServer(
            manager: manager,
            logger: logger,
            logSink: { line in logger.info("\(line)") }
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
