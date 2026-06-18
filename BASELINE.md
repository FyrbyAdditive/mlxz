# Performance baseline (`mlxz-serve --bench`)

Repeatable, fixed-prompt benchmark for proving perf changes. Run via the `bench` mode of
`mlxz-serve` (reuses the exact server load + `MLXInferenceEngine.generate` MTP path, so every
`EnginePerfOptions` knob is exercised identically). The authoritative per-step decode metric is the
engine's env-gated `[DECODE]` line (STEPWALL ms/step), which `--bench` forces on; the harness also
prints wall-clock TTFT, decode tok/s, and peak GPU memory, median over warm runs.

## How to run

```bash
scripts/build-mlx.sh mlxz-serve
.xcode-build/Build/Products/Debug/mlxz-serve --bench \
  --model mlx-community/Qwen3.6-27B-4bit \
  --mtp-draft mlx-community/Qwen3.6-27B-MTP-4bit \
  --prompt-tokens 512 --bench-max-tokens 256 --iters 3
```

- `--prompt-tokens 512` isolates **decode** (small prompt). Use a large value (~80000) to isolate
  **prefill / TTFT**.
- 1 warmup iteration is discarded; metrics are the median over `--iters` warm runs.
- Acceptance for trusting a result: run-to-run decode variance (stddev/median) **< 3%**.

## Baseline — short prompt (decode-isolating)

Machine: this dev Mac. Model: `Qwen3.6-27B-4bit` + `Qwen3.6-27B-MTP-4bit` drafter, greedy (temp=0),
kvBits=4, gpuCacheMB=512, prefixCacheSlots=16. promptTokens≈512, maxTokens=256, 3 warm runs.

| Metric | Value |
| --- | --- |
| decode tok/s (wall) | **27.20** (stddev/median 0.4%) |
| STEPWALL ms/step (`[DECODE]`) | **65.9–67.0** (backbone ~62, mtp ~3.6, unaccounted ~0.4) |
| TTFT (warm, 512 tok) | 0.59 s |
| decode time (256 tok) | 9.37 s |
| peak GPU memory | 19.00 GB |

Notes:
- Decode is **memory-bandwidth-bound**: the backbone matmul over the 16 GB of 4-bit weights dominates
  STEPWALL; per-token CPU work is ~0.4 ms (unaccounted).
- MTP is accepting ~1.8 tokens/step (142 steps → 256 tokens), the speculative speedup at work.
- Variance is 0.4% — well under the 3% bar, so a ≥3–5% change is a real signal.

## Experiment ledger (proven wins fold into the baseline; failures recorded as no-ops)

| Phase | Lever | Result | Decision |
| --- | --- | --- | --- |
| 0 | Benchmark harness | variance 0.4% < 3% | accepted (this baseline) |
| 1A | Wired memory limit (20 GB) | decode 27.20→27.02 tok/s (−0.7%, noise); variance 0.4%→2.5%; peak 19.0 GB unchanged | **no-op at idle** — kept as off-by-default `--wired-mb` (the 19 GB working set already fits the device's recommended working set, so nothing was paging; valuable only under genuine memory pressure). Default stays OFF. |
| 1B | GPU cacheLimit sweep | tok/s by cacheMB: 0→26.09, 256→26.16, **512→27.20**, 1024→21.21, 2048→21.13 (peak 19.0 GB at all) | **current default (512) is optimal** — no change. Larger caches **regress −22%** (eviction/thrash vs resident weights); smaller churn −4%. Confirms the 512 default + documents a guardrail against raising it. |
| 1B | GPU cacheLimit sweep | _pending_ | — |
| 1C | Adaptive auto-sizing | _pending_ | — |
| 2 | GPU profiling gate | _pending_ | — |
| 3 | mx.compile sub-block | _pending_ | — |
