import Foundation
import Hummingbird
import Logging
import MLXZCore

/// Configuration for the HTTP server.
public struct ServerConfig: Sendable {
    public var host: String
    public var port: Int
    /// When set, requests must present `Authorization: Bearer <apiKey>`.
    public var apiKey: String?

    public init(host: String = "127.0.0.1", port: Int = 8080, apiKey: String? = nil) {
        self.host = host
        self.port = port
        self.apiKey = apiKey
    }
}

/// The OpenAI-compatible HTTP server. Owns a Hummingbird app run inside a child task,
/// reading the currently-loaded model from the shared `ModelManager` per request.
public actor InferenceServer {
    private let manager: ModelManager
    private let gate: GenerationGate
    private let logger: Logger
    /// Optional sink for human-readable log lines surfaced in the UI.
    private let logSink: (@Sendable (String) -> Void)?

    private var runTask: Task<Void, any Error>?
    private(set) public var isRunning = false

    public init(
        manager: ModelManager,
        maxConcurrent: Int = 1,
        logger: Logger = Logger(label: "mlxz.server"),
        logSink: (@Sendable (String) -> Void)? = nil
    ) {
        self.manager = manager
        self.gate = GenerationGate(maxConcurrent: maxConcurrent)
        self.logger = logger
        self.logSink = logSink
    }

    public func start(_ config: ServerConfig) async throws {
        guard !isRunning else { return }

        let router = RouterBuilder(
            manager: manager,
            gate: gate,
            apiKey: config.apiKey,
            logSink: logSink
        ).build()

        let app = Application(
            router: router,
            configuration: .init(
                address: .hostname(config.host, port: config.port),
                serverName: "mlxz"
            ),
            logger: logger
        )

        let task = Task {
            try await app.runService()
        }
        runTask = task
        isRunning = true
        logSink?("Server listening on http://\(config.host):\(config.port)")
        logger.info("server started", metadata: ["host": .string(config.host), "port": .stringConvertible(config.port)])
    }

    public func stop() async {
        runTask?.cancel()
        runTask = nil
        isRunning = false
        logSink?("Server stopped")
        logger.info("server stopped")
    }
}
