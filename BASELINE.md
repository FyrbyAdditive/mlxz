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
| 1C | Adaptive auto-sizing | — | **skipped**: 1A/1B were no-ops (512 already optimal, wired no help), so there's no hand-tuned winner to auto-reproduce; would add complexity for ~0 gain. |
| 2 | GPU profiling gate | decode `unaccounted ≈ 0 ms` (−0.9 ms, noise) → no dispatch overhead; cold prefill 14.8 s / 12k tok = matmul-bound; MoE/MLP already fused (`SwitchGLU`/`quantizedMatmul`, `scaledDotProductAttention`), only glue is `silu*` (negligible) | **gate NOT met** — decode & prefill are matmul/bandwidth-bound with ~0% fusible/dispatch headroom. |
| 3 | mx.compile sub-block | not pursued | **dropped** — Phase 2 gate (>10% fusible/dispatch share) not met; documented dead end. |
| K1 | bd=256 fused steel attention (gate relax) | 24k −6% / 48k +2–5% (regression); root cause: 4-bit KV uses un-fused `quantizedScaledDotProductAttention`, not steel | **reverted** — no win; real gap is the quantized-KV attention path (see below) |
| K2 | fused flash-attention for quantized-KV path | from-scratch kernel BUILT (correct, memory-safe) but no perf win (per-element reductions lose to NAX-GEMM): TTFT 2k +21% / 8k par / 24k −5% / 48k +4% | kernel **reverted**; but the investigation found + fixed a real causal-mask correctness bug (K3) |
| K3 | causal-mask leak fix (`leastNormalMagnitude`→`-greatestFiniteMagnitude`) in `quantizedScaledDotProductAttention` | the bool/causal mask used +1.18e-38 (leaks `exp(-max)` weight onto masked/future keys); perf-neutral (8k TTFT/decode unchanged), greedy output coherent | **LANDED** — genuine correctness fix on the production quantized-KV attention path |
| O1 | GPU cache-limit re-test full-RAM (omlx sets cache=RAM) | decode tok/s by gpuCacheMB: **512→27.3**, 4096→21.1, 16384→21.2, 49152→21.3 (all var ≤0.2%). Order-controlled (ran 512 both first AND last → 27.3/27.1; 16384 first AND mid → 21.3/21.2) so cache value is causal, not thermal/order. | **no change — 512 confirmed optimal.** omlx's full-RAM advice is HARMFUL here: a big cache hoards scratch buffers that evict the resident 16GB weights (backbone 65.7→84ms, −22% decode). We never clear_cache mid-decode, so the omlx cache+clear pairing doesn't apply. |
| O2 | Predictive prefill chunk-shrink (omlx guards async Metal OOM at long ctx) | measured cold-prefill peak memory (MLX-reported): 96k→34.25GB, **128k→38.67GB** (RSS 15GB), both EXIT=0, no OOM/SIGABRT. 64GB machine → ~25GB headroom even at the user's 128k ceiling. | **not needed — not implemented.** Chunked prefill (512-tok chunks, eval per chunk) already bounds transient memory; 128k peaks at 38.67/64GB. The guard would prevent a non-problem. (The earlier 120GB crash was the buggy qflash kernel, not normal prefill.) NB: 128k TTFT is 227s — the real long-ctx cost, targeted by O4/SpecPrefill, not memory. |
| O3 | Chain-hashed block-level prefix matching (omlx cross-request block dedup) | two distinct conversations sharing a ~1900-tok system prompt: convo B `[PREFIX] reused=1536 fresh=352` — reused 81% from convo A's snapshot (block-aligned at 3×512, inside the shared region). | **not needed — not implemented.** Our cold-multi-boundary capture (commit 14f650c) + linear `reuseCount` over the LRU **already does cross-conversation block-level reuse**. chain-hash only adds O(n/block) lookup (negligible at 16 entries); reuse is already firing, so no TTFT to recover. |
| O4 | SpecPrefill (omlx: small draft model scores token importance → sparse-prefill target) | code-read of omlx `specprefill.py`: scoring **requires a full prefill of a model to get keys**, then N=8 lookahead + sparse re-prefill. Its speedup depends entirely on the scorer being a **separate small draft model**. | **not viable as designed — not implemented.** We have NO separate small scorer (our MTP drafter is a 1-layer fused head, not a standalone prefill model). Scoring on the 27B target = the full 227s prefill we're trying to avoid → net loss. A separate small model as scorer would be a new dependency + unvalidated cross-model importance transfer through our 48 GDN layers + the unresolved GDN-on-pruned-tokens correctness risk — a large speculative project, not a port. Documented; 227s 128k-TTFT remains compute-bound and is hardware-bound (M5 prefill already 3.46× NAX-accelerated). |

## Metal-4 / NAX tensor-op matmul (M5 Max investigation)

This machine: **Apple M5 Max** (`applegpu_g17s`, gen 17), macOS 26.5.1, Metal 4, SDK 26.5. MLX 0.31.1
ships a Metal-4 NAX tensor-op matmul path (`mpp::tensor_ops::matmul2d`, 16×16 fragments) gated by
`is_nax_available()` (needs gen ≥17 + macOS ≥26.2 — both hold here). NAX kernels are JIT-compiled at
runtime (not in the metallib). A/B via `MLX_METAL_GPU_ARCH=applegpu_g16s` (forces the older
`simdgroup_matrix<8,8>` MMA fallback):

| Scenario | NAX-on (g17s, default) | Forced-MMA (g16s) | NAX win |
| --- | --- | --- | --- |
| **Cold prefill TTFT** (~12k tok) | **14.98 s** | 51.83 s | **3.46× faster** |
| Decode tok/s (512-tok prompt) | 27.39 (var 0.3%) | 26.10 (var 2.5%) | ~5% + much stabler |
| Peak memory | 19–23 GB | same | — |

**NAX is ALREADY ACTIVE and delivering** — a 3.46× prefill speedup (matches Apple's "3.5–4× prefill on
M5" claim) and a smaller ~5% decode gain (decode is bandwidth-bound). There is **no gap to patch**: the
M5 hardware tensor-op path engages by default. (Caveat: spoofing the arch also flips the minor
`qmv_batch_limit` gen tuning, a small confound in the decode delta; the prefill 3.46× is unambiguous.)

### mlx-swift 0.31.1 → 0.31.4 bump

Bumped the pin (`.upToNextMinor(from: "0.31.3")` in the MTP fork already allowed it; the resolution was
stale at 0.31.1). 0.31.2–0.31.4 are **patch-only** (fmt 12.1, mxfp4 non-affine quant fix, more
`compile()` overloads up to arity 8/4, eval-lock safety) — **no kernel/perf changes**. Benchmarked:

| Scenario | 0.31.1 | 0.31.4 | Δ |
| --- | --- | --- | --- |
| Decode tok/s (short) | 27.20 | 26.91 | −1.1% (noise) |
| Cold prefill TTFT (16k) | 14.98 s | 15.91 s | +6% (single-shot noise) |
| Peak memory | 19.0 GB | 19.0 GB | unchanged |

**Performance-neutral** (deltas within run-to-run noise; no decode-path changes by construction).
**Kept** (user decision) for the correctness/safety fixes and staying current — not for speed. Note:
upstream left the embedded `MLX_VERSION`/`version.h` string at "0.31.1" through these tags (the git tag
is 0.31.4; the version constant lagged upstream).

## Per-kernel analysis (code-level read of the actual .metal kernels)

Model is a **dense hybrid** (verified config.json): hidden 5120, head_dim 256, 24 q / 4 kv heads,
**16 full-attention + 48 linear-attention (GatedDeltaNet) layers** of 64. Per-kernel decode/prefill map
and the micro-optimization hypotheses tested:

- **4-bit quantized matmuls (qmv_fast / qmm_nax)** — every layer's q/k/v/o + MLP projections; the
  bandwidth-bound dominant cost. Decode (M≤2, incl. MTP verify) → `qmv_fast`; prefill (M=512 chunks) →
  `qmm_nax` (NAX). **Two dispatch micro-knobs checked and both NO-OP for our shapes:**
  - *qmv fast-path gate* `K % 512 == 0`: all our Ks (5120/6144/17408) are %512==0 → already `qmv_fast`,
    never the slow bounds-checked `qmv`. No gate-miss.
  - *qmv→qmm batch-limit* (`get_qmv_batch_limit`, `quantized.cpp:84`): M5 (gen 17, arch 's') hits the
    `else`/`default` branch = **10** for our D/O>4096. Our workload is M≤2 (decode/MTP) or M=512
    (prefill) — neither lands in the 3–10 window the limit governs, so it doesn't affect us.
- **GatedDeltaNet recurrence (48/64 layers)** — a **custom fused JIT Metal kernel in the fork**
  (`MLXVLM/Models/Qwen35.swift:160`, `gated_delta_step`): one SIMD group per (batch,value-head), the
  time loop runs in-kernel with SSM state in registers, two `simd_sum` reductions/step; grid
  `(32, Dv=128, B*Hv)`, threadgroup `(32,4,1)`. Already fused (no per-timestep CPU dispatch) and
  operates on tiny per-token state, not the 16 GB weights — secondary to the matmul bandwidth cost.
- **Attention (16/64 layers)** — decode: head_dim 256 ∈ vector-SDPA set {64,96,128,256} → specialized
  `sdpa_vector_*_256_256` (fine). **Prefill: head_dim 256 ∉ steel allow-list {64,80,128}** (`scaled_
  dot_product_attention.cpp:622`; steel kernels only instantiated for bd∈{64,80,128}) → falls to the
  **un-fused** QK^T+softmax+SV path (the matmuls still get NAX). This is the one real code gap.

**Sizing the attention gap (data-driven, prefill TTFT vs context length):**

| Context (tok) | Cold TTFT | matmul O(L) share | attention O(L²) share |
| --- | --- | --- | --- |
| ~1,500 | 1.92 s | 90% | ~1% |
| ~6,000 | 7.25 s | 96% | ~5% |
| ~18,000 | 24.13 s | 86% | ~14% |
| 32,000 (fit) | ~47 s | 78% | ~22% |
| 64,000 (fit) | ~116 s | 64% | ~36% |

Fit `TTFT = 1.16 ms/tok·L + 1.0e-8 s·L²`. Prefill is **O(L) matmul-dominated** at normal contexts;
the un-fused attention only becomes significant at **very long contexts (≥32k)**.

### Attempt: fused head_dim-256 attention kernel (TRIED, then REVERTED — measured no-win)

The user runs long contexts (up to 128k), so the head_dim-256 fused-attention gap was worth attacking.
Findings:
- **bd=256 is feasible**: the base steel kernel is bd-generic (JIT-compiled on-demand via
  `get_template_definition`; macOS build uses `jit_kernels.cpp`, not the precompiled metallib), fits
  threadgroup memory at bq=32/bk=16 (~29 KB < 32 KB), and the NAX kernel is bd-templated
  (`TD = BD/UD = 8`). Relaxing `sdpa_full_supported_head_dim` to include 256 → the bd=256 kernel
  **compiled and ran with no crash/JIT error**.
- **But it produced no win** (controlled A/B, same binary ± the gate, cold prefill):

  | Context | 4-bit-KV unfused (baseline) | bd=256 fused | fp16-KV + bd=256 fused |
  | --- | --- | --- | --- |
  | 24k | 24.9 s (median/3, 1.8% var) | 23.3 s (−6%) | 23.4 s (−6%) |
  | 48k | 53.7 s | 54.9 s (**+2%**) | 56.3 s (**+5%**) |

  Break-even at 24k, **regression at 48k** — fails the ≥10% bar. **Reverted** (MLX checkout clean).

### ROOT CAUSE (the related issue that prevents the improvement)

With the default **4-bit KV cache**, the 16 full-attention layers do **NOT** use the steel/`MLXFast`
fused SDPA at all. `maybeQuantizeKVCache` (MTPSession.swift:266) makes the cache a `QuantizedKVCache`,
so `attentionWithCacheUpdate` (AttentionUtils.swift:57) routes to **`quantizedScaledDotProductAttention`**
(KVCache.swift:1933) — a **separate, un-fused implementation** that **materializes the full S = QK^T
scores matrix** (`quantizedMM` → `MLX.where` causal mask → softmax → `quantizedMM`·V). The
`sdpa_full_supported_head_dim` gate governs a path our prefill never reaches with quantized KV → the
bd=256 change was inert. Even forcing fp16 KV (so the fused steel path IS used) gave no consistent
long-context win — the bd=256 tile (doubled head-dim footprint, bq=32/bk=16) doesn't beat the NAX-GEMM
unfused path at these shapes; occupancy loss offsets the no-materialize-S benefit.

**The genuine long-context attention gap is therefore NOT "add bd=256 to steel" — it is that there is
no fused (flash-attention-style) kernel for the quantized-KV path.** Closing it means writing a
quantized flash-attention kernel (large new-kernel effort, high risk) or accepting fp16 KV (no measured
win here, +KV memory). Both are out of scope as measured no-wins; documented so the gap isn't
re-attempted blindly. Decode is unaffected (head_dim 256 uses the specialized vector-256 kernel).

### K2: resolving the quantized-KV attention gap (investigated deeply, no win landed)

MLX 0.31.4 has **no fused quantized-SDPA** (no Metal kernel, no C++/Swift API). The fork's
`quantizedScaledDotProductAttention` (the only quantized-KV attention) is un-fused: `quantizedMM`
(QK^T, transpose=true → **NAX**) → `MLX.where` causal mask → `softmax` → `quantizedMM` (·V,
transpose=false → **non-NAX**). Three angles investigated to close the gap, all measured:

1. **Force fp16 KV → use the fused steel SDPA** (`--kv-bits 0` + bd=256 gate). 48k: 56.3 s vs 53.7 s
   baseline → **+5% regression**. The fused flash kernel (no S materialization) does **not** beat the
   un-fused NAX-GEMM path at our shapes; bd=256 tile occupancy loss dominates.
2. **Enable non-transpose NAX for the ·V matmul** (the one attention GEMM that misses NAX, since the
   `qmm` dispatcher gates NAX on `transpose==true`). A complete `qmm_n_nax` kernel exists, so relaxed
   the gate. 24k: −2.6% (noise); **48k: 58.96 s vs 53.7 s → +9.8% regression.** `qmm_n_nax` is
   *slower* than the MMA non-transpose path here — which is **why upstream gates it off**. Reverted.
3. **From-scratch fused quantized flash-attention kernel — BUILT, then REVERTED (no perf win).**
   Wrote a custom `MLXFast.metalKernel` (270 lines): one threadgroup per (batch, q-head, query-tile),
   online-softmax over the KV cache, dequantizing 4-bit K/V tiles in-register, never materializing S.
   - **Correctness: proven** via a standalone tiny-shape self-test — matches a correctly-masked
     reference to fp16 rounding (~1e-3 max-abs diff) for both `.none` and `.causal` masks.
   - **Memory: safe** — bounded in isolation (0.064 GB at Lk=4096) and in-model 2k→48k (peak
     20.5→26.7 GB, RSS ≤15 GB). The earlier 120 GB OOM did **not** reproduce with the corrected,
     opt-in-gated kernel (it was a buggy earlier draft and/or the GUI app running concurrently).
   - **Perf: not a win.** In-model TTFT vs un-fused baseline: 2k +21%, 8k ~par, 24k −5.5%, 48k +3.8%.
     The kernel's per-element `simd_sum` QK/·V reductions lose to MLX's **NAX-GEMM** un-fused path
     (the QK^T `quantizedMM` is `transpose=true` → NAX hardware matmul). Beating it would require a
     simdgroup-matrix/NAX flash kernel (much larger effort). **Reverted.**

   **BUT this investigation found a real correctness bug in the reference** (see fix below).

## Conclusion

The inference stack is already well-tuned for this machine: decode is memory-bandwidth-bound at the
matmul roofline (~0 ms CPU/dispatch overhead per step), the 512 MB GPU cache default is empirically
optimal (raising it regresses 22%), the model's MoE/MLP/attention already route through MLX's fused
fast paths, and the 19 GB working set fits the device's recommended working set (no paging to wire
away). The audit produced **one reusable asset** (the committed `--bench` harness + this ledger, so any
future change is provable) and **two documented guardrails** (don't raise the cache; wired-memory is
off-by-default, useful only under real memory pressure). No accepted speedup — the honest, data-backed
outcome is that the cheap reversible knobs are already at their optimum and the invasive lever has no
headroom to exploit.

The deep per-kernel pass (bd=256 steel, fp16-fused, non-transpose NAX, and a from-scratch fused
quantized flash kernel) produced **no perf win** — MLX's NAX-GEMM attention is already near-roofline at
our shapes — but it **did surface and fix a real correctness bug** (K3): the quantized-KV attention
masked future/padding keys with `+leastNormalMagnitude` instead of a large negative, leaking
`~exp(-max)` attention weight onto masked positions (worse at shorter causal windows). Fixed to
`-greatestFiniteMagnitude` on all three mask sites; perf-neutral, greedy output coherent. **This bug fix
is the one landed change** — proof that "do it properly" pays off even when the headline optimization
turns out to be a no-op.

## Cross-scenario comparison (Metal-4 deep-dive deliverable)

All on Apple M5 Max, greedy, median of warm runs. "MMA" = forced fallback via
`MLX_METAL_GPU_ARCH=applegpu_g16s`.

| Config | Decode tok/s (512-tok) | Cold prefill TTFT (~12k) | Peak mem | Notes |
| --- | --- | --- | --- | --- |
| **0.31.1, NAX (default)** | 27.20 | 14.98 s | 19–23 GB | baseline |
| 0.31.1, forced-MMA | 26.10 | **51.83 s** | same | NAX off → 3.46× slower prefill |
| **0.31.4, NAX (kept)** | 26.91 | 15.91 s | 19–23 GB | perf-neutral vs 0.31.1; current |

**Bottom line for the user to choose from:** the single biggest Metal lever — the M5's Metal-4 NAX
tensor-op matmul path — is **already on and giving a 3.46× prefill speedup + ~5% decode** out of the
box; there is nothing to enable or patch. Bumping mlx-swift to 0.31.4 is perf-neutral and was **kept**
for its correctness/safety fixes. No further Metal/shader change is warranted on this hardware; the
remaining headroom is hardware-bound (already realized by the M5 tensor units). Should this server ever
run on **pre-M5 silicon (gen <17)**, NAX won't engage there and prefill will be ~3.5× slower — the
`--bench` harness will surface that immediately.

---

## Experiment: DSpark speculative decoding (2026-07-07, dspark branch)

DeepSeek DSpark (arXiv:2606.19348) drafters for Qwen3-8B/14B, Swift port in the fork
(MLXLMCommon/DSpark), auto-attach via --dspark-draft. Full evidence ledger:
docs/dspark/findings.md. Outcome summary:

- Correctness: drafter parity EXACT vs the Python reference; greedy divergence 2.04%/token
  vs a measured 2.67%/token no-speculation kernel-shape ceiling (byte-identity across
  M=1/M>1 forwards is unattainable on Metal — 1.0-1.3% of greedy positions are exact bf16
  ties); sampling distribution-exact (chi-square) with teacher-forced logprob parity.
- Performance: WINS on math/code with fp16 KV (sustained 1.08-1.09×, bursty up to 1.29×;
  bf16 targets up to 1.47×); LOSES on quantized KV at ctx ≥ ~2k (M>1 attention cliff) and
  on chat content. Auto-attach therefore requires --kv-bits 0.
- ⚠️ LEDGER CORRECTION: all pre-2026-07 numbers in this file were DEBUG builds. Release:
  27B+MTP decodes 44.4 tok/s (not 27.2), Qwen3-8B-4bit 103.6 tok/s (not ~48). Fast models
  are CPU-bound in Debug. Re-baseline in Release before the next optimization pass.
- Follow-ups: per-row qmv split for quantized small-M verify attention (fixes the cliff,
  also speeds MTP verify); n-gram lookup drafting (the Python port commits ~6 tok/round on
  copy runs); Gemma4 target (needs full-cache + forced-window masks); cap auto-controller.
