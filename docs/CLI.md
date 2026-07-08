# mlxz-serve — command-line usage

`mlxz-serve` is the headless counterpart to the mlxz app: it loads one model and serves
the same OpenAI-compatible API, with no GUI. Useful for scripts, CI, remote sessions, and
benchmarking.

## Build

```bash
scripts/build-mlx.sh mlxz-serve                # Debug (development)
# Release — use this for anything performance-related (Debug halves fast-model decode):
MLXZ_MLX=1 xcodebuild build -scheme mlxz-serve -configuration Release \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath .xcode-build \
  -clonedSourcePackagesDirPath .build/xcode-packages -scmProvider system
```

The binary lands in `.xcode-build/Build/Products/<Configuration>/mlxz-serve`.

## Serve a model

```bash
mlxz-serve --model mlx-community/Qwen3-8B-4bit --port 8080
```

First run downloads the model from Hugging Face. If the model has an official DSpark
drafter (Qwen3 4B/8B/14B, Gemma4-12B), it auto-attaches for a decode speedup; MTP
drafters attach via `--mtp-draft`.

### Core flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--model <repo>` | (required) | Hugging Face repo id to serve |
| `--host` / `--port` | 127.0.0.1 / 8080 | Bind address (use `0.0.0.0` for LAN) |
| `--api-key <key>` | none | Require `Authorization: Bearer <key>` |
| `--max-batch N` | 8 | Concurrent requests decoded together (batchable models) |
| `--max-queue N` | 0 (unbounded) | Waiting-request cap before 429 |
| `--print-copilot-config` | — | Print the VS Code model entry and exit |

### Speculative decoding

| Flag | Default | Purpose |
| --- | --- | --- |
| `--dspark-draft <auto\|off\|repo>` | auto | DSpark drafter policy (auto-resolves official drafters) |
| `--draft-block N` | 3 | Tokens drafted+verified per round (2–3 is the Apple Silicon sweet spot) |
| `--confidence-threshold X` | 0 (off) | Trim low-confidence draft tails before verify |
| `--mtp` / `--no-mtp` | on | Native MTP self-speculative decoding (Qwen3.5/3.6) |
| `--mtp-draft <repo>` | none | Attach a standalone MTP drafter checkpoint |

Environment kill switches (diagnostics): `MLXZ_DSPARK_ADAPTIVE=0` (always draft),
`MLXZ_DSPARK_LOOKUP=0` (no n-gram lookup drafts), `MLXZ_QKV_ROWSPLIT=0` (legacy quantized
attention path).

### Memory & performance

| Flag | Default | Purpose |
| --- | --- | --- |
| `--kv-bits N` | 4 | Quantize the KV cache (0 = full precision) |
| `--max-kv-size N` | unbounded | Cap KV cache tokens (rotating cache) |
| `--prefix-cache` / `--no-prefix-cache` | on | Reuse KV across requests sharing a prompt prefix |
| `--prefix-cache-slots N` | 16 | Prefix-snapshot LRU slots |
| `--prefix-cache-mb N` | 2048 | RAM ceiling for prefix snapshots |
| `--snapshot-block N` | 512 | Snapshot capture granularity (tokens) |
| `--prefill-chunk N` | 512 | Prompt tokens per prefill pass |
| `--gpu-cache-mb N` | 512 | MLX GPU buffer-cache bound |
| `--wired-mb N` | 0 (off) | Wired-memory limit so weights stay resident |
| `--reasoning-budget N` | 2048 | Cap on `<think>` tokens before force-close |

### Diagnostics

Set `MLXZ_DECODE_DIAG=1` for per-request `[DECODE]`/`[SPEC]` stderr telemetry (step
timing, acceptance histogram, adaptive-controller state) and `MLXZ_PREFIX_DIAG=1` for
prefix-cache hit/miss lines.

## Endpoints

- `GET /health`
- `GET /v1/models` — loaded + installed models
- `POST /v1/chat/completions` — streaming + non-streaming, tool calling, image input
- `POST /v1/responses` — structured streaming events, tool calling, image input
- `POST /v1/embeddings`

### Smoke tests

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}]}'

# streaming
curl -sN http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}'

# tool calling
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"weather in Paris?"}],
       "tools":[{"type":"function","function":{"name":"get_weather",
       "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}'
```

## Benchmark modes

These load the model through the exact server path, run, print metrics, and exit:

| Flag | What it measures |
| --- | --- |
| `--bench` | Fixed-prompt decode/TTFT/memory (median over `--iters`, 1 discarded warmup) |
| `--bench-compare` | Order-controlled speculative-vs-plain A/B per prompt suite |
| `--bench-lossless` | Greedy spec-vs-plain divergence vs the `--tie-flip-budget` kernel ceiling |
| `--bench-verify-curve` | Multi-token verify cost at `--verify-ctx` context lengths |

Knobs: `--prompt-tokens` (512 decode-isolating / 80000 prefill-isolating),
`--bench-max-tokens`, `--iters`. Always benchmark **Release** builds on a machine with no
other `mlxz-serve` processes running (`pkill -f mlxz-serve` first) and no other GPU load.
Methodology and the measurement ledger: [`BASELINE.md`](../BASELINE.md) and
[`dspark/findings.md`](dspark/findings.md).

## Using from VS Code Copilot (BYOK)

1. Start the server with a loaded model.
2. In VS Code Insiders, add a **Custom (OpenAI-compatible) endpoint** at
   `http://127.0.0.1:8080/v1`.
3. Paste the entry from `--print-copilot-config` into your `chatLanguageModels.json`.
   It sets `apiType: chat-completions`, `toolCalling: true`, and `vision` per the model.
