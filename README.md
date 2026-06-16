# mlxz

A macOS-native app that serves local **Qwen / Qwen MoE** models (incl. the latest Qwen3.6
family, MoE like Qwen3.6-35B-A3B, and vision models) to local apps over an **OpenAI-compatible
HTTP API** — built for **VS Code Insiders + GitHub Copilot Chat (BYOK)** as the primary consumer.

- Download models from HuggingFace directly in the GUI, load one, start the server.
- `/v1/chat/completions` with **streaming and tool/function calling** (what Copilot agent mode needs).
- Model-agnostic engine and endpoint-agnostic server — adding model families or endpoints
  (responses, embeddings, …) is a small, localized change.

Requires **Apple Silicon** and **macOS 26+** (MLX requirement).

## Architecture

An SPM workspace of library targets keeps the seams honest (the engine module physically can't
import SwiftUI; the server can't import MLX). The concrete MLX engine is injected at the
composition root, so the server and UI are testable against mocks.

```
MLXZCore  ◄── MLXZInference  (the only module that imports MLX*)
   ▲ ▲   ◄── MLXZHub
   │ └─────── MLXZServer  ──► Hummingbird (depends on MLXZCore protocols only)
   └───────── MLXZUI      ──► SwiftUI
App / mlxz-serve ──► everything (composition root only)
```

| Module | Responsibility |
| --- | --- |
| `MLXZCore` | Wire-/engine-independent types (`GenerationRequest`/`GenerationEvent`, `ChatMessage`, `ToolDefinition`/`ToolCall`), the `InferenceEngine` + `ModelLoading` seams, the `ModelManager` actor, `ModelCapabilityDetector`, and a streaming Qwen/Hermes `ToolCallParser`. |
| `MLXZInference` | The only MLX-touching module. `MLXModelLoader` loads via mlx-swift-lm (auto-dispatches LLM/VLM factories → Qwen dense, MoE, VLM all load); `MLXInferenceEngine` maps our request onto `ModelContainer.generate` and translates `Generation` → `GenerationEvent` (native tool calls + a fallback `<tool_call>` parser). |
| `MLXZServer` | Hummingbird server. `OpenAIEndpoint` seam + `RouterBuilder` generic handler (decode → validate caps → translate → stream/non-stream). `ChatCompletionsEndpoint` + SSE chunk encoder, optional API-key middleware, OpenAI-format errors, depth-1 `GenerationGate`. |
| `MLXZHub` | `HubCatalog` (HF `/api/models` search), `LocalModelStore` (enumerate the HF cache), `CopilotConfig` (generate the VS Code `chatLanguageModels.json` entry). |
| `MLXZUI` | `@Observable` `AppModel` + SwiftUI views (model library/search, server control, logs). |
| `mlxz-serve` | Headless executable: composition root for CLI/CI. |
| `App/` | The macOS app: supplies `MLXModelLoader` to `AppModel`; `WindowGroup` + `MenuBarExtra`. |

## Building

MLX-dependent targets (`MLXZInference`, `mlxz-serve`, the app) **must be built with `xcodebuild`** —
`swift build`'s emit-module phase can't thread MLX's transitive C-shim modulemaps, and MLX's Metal
kernels need the Metal toolchain (an upstream limitation; see mlx-swift-lm `CONTRIBUTING.md`). These
targets are gated behind `MLXZ_MLX=1` so they're excluded from the default package graph.

```bash
# Fast unit-test loop for the pure-logic targets (Core, Server, Hub, UI) — no MLX, milliseconds:
swift test

# Build the MLX targets / app via xcodebuild (one-time: downloads the Metal toolchain):
scripts/build-mlx.sh mlxz-serve        # the headless server
MLXZ_MLX=1 xcodegen generate           # (re)generate mlxz.xcodeproj for the GUI app
MLXZ_MLX=1 xcodebuild build -project mlxz.xcodeproj -scheme mlxz \
  -destination 'platform=macOS,arch=arm64' CODE_SIGNING_ALLOWED=NO
```

## Running headless

```bash
# build, then:
.xcode-build/Build/Products/Debug/mlxz-serve --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8088
# print the VS Code Copilot config without loading a model:
mlxz-serve --model mlx-community/Qwen3.6-35B-A3B-MTP-4bit --print-copilot-config
```

Smoke tests:

```bash
curl -s http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"hi"}]}'

curl -sN http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hi"}]}'

# tool calling
curl -s http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"weather in Paris?"}],
       "tools":[{"type":"function","function":{"name":"get_weather",
       "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}'
```

## Using from VS Code Copilot (BYOK)

1. Start the server (GUI: load a model → Start; or `mlxz-serve`).
2. In VS Code Insiders, add a **Custom (OpenAI-compatible) endpoint** pointing at
   `http://127.0.0.1:8080/v1`.
3. Paste the model entry from the app's **Server** tab (or `--print-copilot-config`) into your
   `chatLanguageModels.json`. It sets `apiType: chat-completions`, `toolCalling: true`, and `vision`
   per the model. The model then appears in the Copilot model picker and works in ask + agent modes.

## Endpoints

- `GET /health`
- `GET /v1/models` — loaded + installed models
- `POST /v1/chat/completions` — streaming + non-streaming, tool calling, image input
- `POST /v1/responses` — structured streaming events + non-streaming, tool calling, image input

## Status

**Phase 1 + 2 complete**, verified end-to-end against a real model:

- Chat Completions and Responses APIs (both streaming + non-streaming + tool calling).
- Vision input (`image_url` remote or base64 `data:` URL) → VLM image path.
- `/v1/models`, HF catalog search, install enumeration, explicit downloads with progress.
- GUI (model library/search/download, server control, logs) + menu bar.

Known limitation: MTP/speculative decoding is a present-but-inert seam — mlx-swift-lm 3.31.3 has
no native single-model MTP via `ModelContainer.generate`, so speculative requests use standard
decoding (capability is still advertised). Planned next: embeddings (`MLXEmbedders`), legacy
`/v1/completions`, draft-model speculative decoding, prefix/KV-cache reuse. See
`.claude/plans/we-are-building-a-lovely-tiger.md`.
