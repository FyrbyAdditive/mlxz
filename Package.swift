// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "MLXZKit",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXZCore", targets: ["MLXZCore"]),
        .library(name: "MLXZInference", targets: ["MLXZInference"]),
        .library(name: "MLXZServer", targets: ["MLXZServer"]),
        .library(name: "MLXZHub", targets: ["MLXZHub"]),
        .library(name: "MLXZUI", targets: ["MLXZUI"]),
        .executable(name: "mlxz-serve", targets: ["mlxz-serve"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.25.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.8.2"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.6.0"),
    ],
    targets: [
        // MARK: - Core: pure domain types + protocols. No MLX / Hummingbird / SwiftUI.
        .target(
            name: "MLXZCore",
            dependencies: [
                .product(name: "Logging", package: "swift-log"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Inference: the only module that imports MLX*.
        .target(
            name: "MLXZInference",
            dependencies: [
                "MLXZCore",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Server: Hummingbird + OpenAI translation. Depends on Core protocols only.
        .target(
            name: "MLXZServer",
            dependencies: [
                "MLXZCore",
                .product(name: "Hummingbird", package: "hummingbird"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Hub: HuggingFace catalog search + download + local model enumeration.
        .target(
            name: "MLXZHub",
            dependencies: [
                "MLXZCore",
                .product(name: "Logging", package: "swift-log"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - UI: SwiftUI views + view models.
        .target(
            name: "MLXZUI",
            dependencies: [
                "MLXZCore",
                "MLXZHub",
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Headless executable for CLI / CI smoke tests (no SwiftUI).
        .executableTarget(
            name: "mlxz-serve",
            dependencies: [
                "MLXZCore",
                "MLXZInference",
                "MLXZServer",
                "MLXZHub",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Tests
        .testTarget(
            name: "MLXZCoreTests",
            dependencies: ["MLXZCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "MLXZServerTests",
            dependencies: [
                "MLXZServer",
                "MLXZCore",
                .product(name: "HummingbirdTesting", package: "hummingbird"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "MLXZHubTests",
            dependencies: ["MLXZHub", "MLXZCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "MLXZInferenceTests",
            dependencies: ["MLXZInference", "MLXZCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
