// swift-tools-version: 6.2
import PackageDescription
import Foundation

// MLX-dependent targets (MLXZInference, the mlxz-serve executable, and their deps) can only be
// built with xcodebuild — `swift build`'s emit-module phase cannot thread MLX's transitive C-shim
// modulemaps, and MLX's Metal kernels need the Metal toolchain (see mlx-swift-lm CONTRIBUTING.md).
//
// So we gate them behind MLXZ_MLX=1. Default `swift build`/`swift test` covers the pure-logic
// targets (Core, Server, Hub, UI) for a fast unit-test loop; `scripts/build-mlx.sh` sets the flag
// and builds the full graph via xcodebuild.
let includeMLX = ProcessInfo.processInfo.environment["MLXZ_MLX"] == "1"

var products: [Product] = [
    .library(name: "MLXZCore", targets: ["MLXZCore"]),
    .library(name: "MLXZServer", targets: ["MLXZServer"]),
    .library(name: "MLXZHub", targets: ["MLXZHub"]),
    .library(name: "MLXZUI", targets: ["MLXZUI"]),
]

var dependencies: [Package.Dependency] = [
    .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.25.0"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.8.2"),
    .package(url: "https://github.com/apple/swift-log.git", from: "1.6.0"),
]

let v6: [SwiftSetting] = [.swiftLanguageMode(.v6)]

var targets: [Target] = [
    // MARK: - Core: pure domain types + protocols. No MLX / Hummingbird / SwiftUI.
    .target(
        name: "MLXZCore",
        dependencies: [.product(name: "Logging", package: "swift-log")],
        swiftSettings: v6
    ),
    // MARK: - Server: Hummingbird + OpenAI translation. Depends on Core protocols only.
    .target(
        name: "MLXZServer",
        dependencies: ["MLXZCore", .product(name: "Hummingbird", package: "hummingbird")],
        swiftSettings: v6
    ),
    // MARK: - Hub: HuggingFace catalog search + download + local model enumeration.
    .target(
        name: "MLXZHub",
        dependencies: ["MLXZCore", .product(name: "Logging", package: "swift-log")],
        swiftSettings: v6
    ),
    // MARK: - UI: SwiftUI views + view models. Drives Core/Server abstractions; the concrete
    // MLX loader is injected at the App composition root (so the UI stays swift-build-testable).
    .target(
        name: "MLXZUI",
        dependencies: ["MLXZCore", "MLXZHub", "MLXZServer"],
        swiftSettings: v6
    ),
    // MARK: - Tests (pure-logic, run under `swift test`)
    .testTarget(name: "MLXZCoreTests", dependencies: ["MLXZCore"], swiftSettings: v6),
    .testTarget(
        name: "MLXZServerTests",
        dependencies: ["MLXZServer", "MLXZCore", .product(name: "HummingbirdTesting", package: "hummingbird")],
        swiftSettings: v6
    ),
    .testTarget(name: "MLXZHubTests", dependencies: ["MLXZHub", "MLXZCore"], swiftSettings: v6),
]

if includeMLX {
    dependencies += [
        // Our fork of mlx-swift-lm with native MTP speculative decoding. Pinned to an exact revision
        // so external checkouts build the same code (the fork has no release tags of its own — its
        // tags mirror upstream). For local fork development, swap this for:
        //   .package(name: "mlx-swift-lm", path: "../mlx-swift-lm-mtp"),
        .package(
            name: "mlx-swift-lm",
            url: "https://github.com/FyrbyAdditive/mlx-swift-lm-mtp.git",
            revision: "7f20baacb68e213ad74cfbb691793efca48f7698"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.3"),
    ]
    products += [
        .library(name: "MLXZInference", targets: ["MLXZInference"]),
        .executable(name: "mlxz-serve", targets: ["mlxz-serve"]),
    ]
    targets += [
        // MARK: - Inference: the only module that imports MLX*.
        .target(
            name: "MLXZInference",
            dependencies: [
                "MLXZCore",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: v6
        ),
        // MARK: - Headless executable for CLI / CI smoke tests (no SwiftUI).
        .executableTarget(
            name: "mlxz-serve",
            dependencies: [
                "MLXZCore", "MLXZInference", "MLXZServer", "MLXZHub",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: v6
        ),
        .testTarget(name: "MLXZInferenceTests", dependencies: ["MLXZInference", "MLXZCore"], swiftSettings: v6),
    ]
}

let package = Package(
    name: "MLXZKit",
    platforms: [.macOS(.v26)],
    products: products,
    dependencies: dependencies,
    targets: targets
)
