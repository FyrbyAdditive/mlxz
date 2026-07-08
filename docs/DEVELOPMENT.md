# mlxz developer guide

Technical documentation for building, extending, and benchmarking mlxz. End users should
start with the [README](../README.md); CLI usage lives in [CLI.md](CLI.md).

## Architecture

An SPM workspace of library targets keeps the seams honest (the engine module physically
can't import SwiftUI; the server can't import MLX). The concrete MLX engine is injected at
the composition root, so the server and UI are testable against mocks.

```
MLXZCore  ◄── MLXZInference  (the only module that imports MLX*)
   ▲ ▲   ◄── MLXZHub
   │ └─────── MLXZServer  ──► Hummingbird (depends on MLXZCore protocols only)
   └───────── MLXZUI      ──► SwiftUI
App / mlxz-serve ──► everything (composition root only)
```

| Module | Responsibility |
| --- | --- |
| `MLXZCore` | Wire-/engine-independent types (`GenerationRequest`/`GenerationEvent`, `ChatMessage`, `ToolDefinition`/`ToolCall`), the `InferenceEngine` + `ModelLoading` seams, the `ModelManager` actor, `ModelCapabilityDetector`, `DrafterPairing`, and streaming output parsers. |
| `MLXZInference` | The only MLX-touching module. `MLXModelLoader` loads via mlx-swift-lm (auto-dispatches LLM/VLM factories) and attaches speculative drafters; `MLXInferenceEngine` routes requests to the `SpeculativeScheduler` (MTP/DSpark), `BatchGenerationEngine`, `PlainScheduler`, or `cachedStream`; `SnapshotLRU`/`PromptCacheBox` implement prefix-cache reuse; `VerifyCurveBench` is the kernel microbench. |
| `MLXZServer` | Hummingbird server. `OpenAIEndpoint` seam + generic handler (decode → validate caps → translate → stream). Chat Completions/Responses/Embeddings endpoints, SSE encoding, API-key middleware, `GenerationGate` backpressure. |
| `MLXZHub` | HF catalog search, `LocalModelStore` (HF cache enumeration), `CopilotConfig` generation. |
| `MLXZUI` | `@Observable` `AppModel` + SwiftUI views (model library, server control, playground, performance settings, logs). |
| `mlxz-serve` | Headless executable: composition root for CLI/CI. |
| `App/` | The macOS app: supplies `MLXModelLoader` to `AppModel`; `WindowGroup` + `MenuBarExtra`; icon + About assets. |

## Building

MLX-dependent targets (`MLXZInference`, `mlxz-serve`, the app) **must be built with
`xcodebuild`** — `swift build`'s emit-module phase can't thread MLX's transitive C-shim
modulemaps, and MLX's Metal kernels need the Metal toolchain (upstream limitation; see
mlx-swift-lm `CONTRIBUTING.md`). These targets are gated behind `MLXZ_MLX=1` so the
default package graph stays pure.

```bash
# Fast unit-test loop for the pure-logic targets (Core, Server, Hub, UI) — no MLX:
swift test

# Headless server (Debug):
scripts/build-mlx.sh mlxz-serve

# GUI app:
MLXZ_MLX=1 xcodegen generate          # (re)generate mlxz.xcodeproj from project.yml
MLXZ_MLX=1 xcodebuild build -project mlxz.xcodeproj -scheme mlxz \
  -configuration Release -destination 'platform=macOS,arch=arm64'
```

**Always benchmark Release builds.** Debug (`-Onone`) halves fast-model decode throughput
(measured: Qwen3-8B-4bit 48 → 103.6 tok/s; even the 27B was CPU-bound in Debug).

## The mlx-swift-lm fork

mlxz depends on a fork of mlx-swift-lm
([FyrbyAdditive/mlx-swift-lm-mtp](https://github.com/FyrbyAdditive/mlx-swift-lm-mtp)),
pinned by revision in `Package.swift`. For fork development, swap the pin for
`.package(name: "mlx-swift-lm", path: "../mlx-swift-lm-mtp")` (comment in
`Package.swift`), and restore the pin (after pushing the fork) before merging.

The fork adds, on top of upstream:

- **Native Qwen MTP** self-speculative decoding (`MTPSession`, `MTPSpeculativeModel`) —
  drafts one token per step, verified in the same backbone pass.
- **DSpark speculative decoding** (`Libraries/MLXLMCommon/DSpark/`) — DeepSeek's
  semi-autoregressive drafter (arXiv:2606.19348): cross-attention drafter over target
  hidden-state taps, rank-256 Markov head, confidence head, block verify. Includes the
  adaptive draft on/off controller, n-gram lookup drafting, and the quantized-attention
  small-M row split (fixes a 2.5–7× multi-token-verify kernel cliff; also speeds MTP).
- **Gemma4 support** incl. `gemma4_unified` text-only serving and DSpark target taps with
  forced sliding-window masks (rotating caches can't be trimmed for verify rollback).
- Fused GatedDeltaNet prefill kernel in the VLM path (11× cold-prompt TTFT on the 27B).

## Speculative decoding overview

Both MTP and DSpark run as steppable sessions under one fair `SpeculativeScheduler`
(mlxz side), which interleaves requests one step at a time and manages the prefix-snapshot
LRU. Draft tokens are always verified by the target model — greedy output token-matches a
valid greedy decode (byte-identity across kernel shapes is physically unattainable on
Metal: ~1% of greedy positions are exact bf16 ties; see the findings ledger), and
temperature sampling is distribution-exact via rejection sampling.

Key session mechanics (fork, `DSparkSession`): chunked tapped prefill → per round: block
draft → single verify forward (whose hidden-state taps feed the drafter's context for
free) → trim rejected suffix → commit. Whole-generation snapshots + common-prefix
trim-restore give agentic turns TTFT parity with plain decode.

Performance envelope, gates, and the measurement methodology (thermal asymmetry, tie-flip
budgets, kernel-shape probes) are documented in [`dspark/findings.md`](dspark/findings.md)
and [`BASELINE.md`](../BASELINE.md); reproduction scripts in `scripts/dspark/`.

## Testing

- `swift test` — pure-logic suites (Core/Server/Hub/UI), milliseconds.
- Fork tests: `xcodebuild test -scheme mlx-swift-lm-Package -only-testing:MLXLMTests/...`
  — includes DSpark drafter checkpoint-parity gates (set `TEST_RUNNER_MLXZ_DSPARK_CHECKPOINT`
  and `TEST_RUNNER_MLXZ_DSPARK_FIXTURE`; fixtures via `scripts/dspark/export_parity_fixture.py`).
- End-to-end gates: `mlxz-serve --bench-lossless` (correctness) and `--bench-compare`
  (order-controlled perf A/B); see [CLI.md](CLI.md#benchmark-modes).

## Release checklist

1. `swift test` green; fork test suite green.
2. `--bench-lossless` within the tie budget on a supported model; `--bench` on the 27B+MTP
   control within noise of the ledger.
3. `Package.swift` pin points at a pushed fork revision (not the path dependency).
4. `MLXZ_MLX=1 xcodegen generate` + Release `xcodebuild` for the app.
