# DSpark findings ledger (M5 Max, 2026-07)

## Follow-up #1 — small-M row split for quantized attention (2026-07-07, fork 9bada99)

The M>1 quantized-attention cliff is FIXED: M≤8 causal rows now run as independent
single-row attentions (SDPA rows are independent; causality via per-row key-prefix
slices), each on the same fast qmv path plain decode uses. Exact math (parity-tested vs
the unsplit path and an fp16 dequantized reference; kill switch MLXZ_QKV_ROWSPLIT=0).

Verify-cost curve, Qwen3-8B-4bit, kvBits=4 (was → now):
- ctx 512:  M=2 17.1 → 12.9ms · M=4 21.2 → 17.9ms
- ctx 8192: M=2 49.7 → 16.4ms (3.0×) · M=4 55.3 → 24.4ms (2.3×)
- ctx 32768: M=2 137.4 → 25.6ms (5.4×) · M=4 169.9 → 41.5ms (4.1×)
Quantized-KV multi-token verify now costs the same as fp16 KV (keep the 4× memory win).

Effects on the DEFAULT config (kvBits=4, Qwen3-8B, drafter attached):
- Bursty: chat 0.96× / code **1.13×** / math **1.20×** (was 1.02–1.06×).
- Sustained (order-controlled): chat 0.876× / code 0.943× / math 1.012× — the remaining
  gap is the ALU-bound drafter under sustained-load clock drop + low chat acceptance
  (adaptive draft on/off controller = follow-up #2).
- Agentic 3.2k ctx: 0.6× → ~0.72× of plain (verify 30 → 21ms/round; drafter's own ctx
  attention now dominates at 9.1ms/round).
- Divergence rate 2.50%/token (within the 3.00% ceiling budget). NOTE: a same-day
  "0.00% divergence" measurement was invalid — the kvBits auto-gate had silently
  disabled DSpark, so both arms were plain decode. Always confirm [SPEC] lines exist.
- MTP control after the split (it shares the verify path): 44.81 tok/s — no regression.
- Auto-attach re-enabled for all configs (the kvBits gate is obsolete).

## M4/M5 — performance envelope + regression gates (2026-07-07)

### Headline (order-controlled `--bench-compare`, cap 3, 5 paired iters, Release, clean process table)

| config | chat | code | math |
|---|---|---|---|
| Qwen3-8B-4bit, kvBits=4 (default) | 0.82× | 0.90× | 0.95× |
| Qwen3-8B-4bit, fp16 KV | 0.90× | 0.99× | **1.09×** |
| Qwen3-14B-4bit, fp16 KV | 0.89× | 1.02× | **1.08×** |

Bursty/interactive use (cool GPU, per-prompt alternation) is better: 8B fp16 KV measured
1.05×/1.16×/1.29× (chat/code/math); bf16 targets 1.24×/1.43×/1.47× (plain decode is
memory-bound there, so verify headroom is large).

### Why sustained ≠ burst: asymmetric thermal sensitivity

Spec STEPWALL climbs monotonically 24.3ms → 34ms across ~40 consecutive requests while
plain decode stays flat (±1–3%): the drafter+verify rounds are ALU-bound (sensitive to
sustained-load GPU clock drop), plain decode is DRAM-bandwidth-bound (insensitive).
**Order-controlled A/B does NOT cancel thermal drift when arms have asymmetric
sensitivity** — report burst and sustained numbers separately.

### Context envelope (the quantized-KV M>1 cliff in practice)

At ~3.2k-token agentic context with kvBits=4, DSpark decode drops to ~69 tok/s vs plain
~118 tok/s (0.6×) — the M>1 quantized-attention penalty grows with context. Consequence:
**`--dspark-draft auto` only engages when kvBits is off (fp16 KV)**; an explicit drafter
repo forces attach anywhere. Fixing the quantized small-M attention path (per-row qmv
split — SDPA rows are independent) is the follow-up that would reopen the default config
AND speed up MTP verify.

### Regression gates (all PASSED)

- **MTP control (27B + MTP drafter, refactored scheduler)**: 44.39 tok/s (±0.2%) vs the
  documented 27.20 — the delta is the Debug→Release uplift (BASELINE.md numbers were
  Debug builds; the 27B was CPU-bound, not GPU-bound). No regression; ledger refresh due.
- **Agentic prefix reuse with DSpark**: turn-2 reuses the 2560-token snapshot (incl.
  drafter ctx aux slot), TTFT 2.25s → 0.46s. Plain trim-reuse is finer-grained (0.11s) —
  snapshot-boundary granularity, same policy as MTP.
- **Concurrency c=4**: aggregate 99.6 (DSpark) vs 101.2 (plain) tok/s — parity; the
  speculative scheduler additionally gives fair per-request latency (25–29 tok/s each)
  where the plain path serializes FIFO.

### Where DSpark loses (bound the claim)

Chat content (low acceptance ~2.2/step), sustained heavy load, quantized KV at ctx ≥ ~2k,
and very fast baselines (M5 Max plain decode ~100 tok/s leaves ~6ms/round of headroom).
Wins: math/code content, fp16 KV, bursty interactive use, slower/bigger targets, bf16
targets. Same-machine Python-port oracle for context: 1.11–1.21× (8B), 1.16× (14B),
short bursty runs.

## M3 — speculative sampling losslessness (2026-07-07)

Three-layer proof that temperature > 0 output preserves the target distribution:

1. **Sampler exactness (chi-square)**: `SpeculativeVerifierTests.testOutputDistributionMatchesTarget`
   — draft from q, accept w.p. min(1, p/q), residual-resample on reject; the committed
   token's empirical distribution matches p (4k draws × 3 (p,q) pairs incl. adversarial,
   vocab 8, all chi² < 18.475 = df7 @ p=0.01). Hand-computed unit tests cover the
   accept/residual/bonus paths and the identical p/q top-p/top-k truncation.
2. **Live path**: temperature 0.7 → acc/step 2.22; temperature 1.0 + top_p 0.9 → 2.63
   (truncated-q path exercised through the OpenAI endpoint; coherent output).
3. **No quality drift end-to-end**: 12 completions/arm at temp 0.7 (4 prompts × 3),
   teacher-forced under the target: spec −0.4847 [−0.566, −0.400] vs plain −0.4915
   [−0.576, −0.409] mean logprob/token (bootstrap 95% CIs overlap almost entirely).
   Scripts: scripts/dspark/collect_arm.py + score_arms.py.

**Gate: PASSED.**

## M2 — the greedy "losslessness" standard on MLX/Metal (2026-07-07)

Byte-identity between speculative and plain greedy decode is UNATTAINABLE on this stack —
not an implementation property. Evidence chain (all reproducible; probe scripts in
scripts/dspark/):

1. **Exact ties are common**: 1.0–1.3% of greedy decode positions have top-2 logit gap
   < 1e-4 (i.e. literal bf16 ties) on Qwen3-4B-bf16 / Qwen3-8B-4bit.
2. **Kernel numerics depend on forward shape**: replaying a model's OWN greedy tokens
   through M=4-token forwards (pure mlx_lm, no speculation) flips argmax at:
   1.00%/token (bf16 weights), 0.50% (4-bit weights, fp16 KV, n=200),
   **2.67%/token (4-bit weights + 4-bit quantized KV — the mlxz default)**.
3. **DSpark divergence is below that ceiling in every config**: measured spec-vs-plain
   first-divergence rates ~0.8% (bf16), ~1.2% (fp16 KV), 2.04% (4-bit KV).
4. Acceptance is healthy and matches the Python oracle (2.4–3.1 tok/step at cap 3), and
   the M1 parity gate showed exact drafted-token agreement with the reference — the
   implementation is correct; the flips are tie-breaks between two legitimate greedy
   readings of the same model.

**Gate redefinition**: `--bench-lossless` passes when the estimated per-token divergence
rate is within `--tie-flip-budget` (default 0.03, just above the measured ceiling).
The user-facing claim is the field-standard one: output always token-matches a valid
greedy decode of the target (and for temperature > 0, the sampling distribution is
preserved exactly — M3).

## M0 — go/no-go findings (2026-07-06/07)

## Oracle: mlx-dspark (Python) on this machine

Qwen3-8B-4bit target, 4-bit drafter, 256 new tokens, greedy (two runs; the Python port
does not order-control arms, so treat the ratio spread as its noise floor):

| cap | accept/round | run A | run B |
|-----|-------------|-------|-------|
| baseline | 1.00 | 105.5 tok/s | 106.6 tok/s |
| 1 | 1.79 | 1.07× | 1.07× |
| **2** | **2.29** | **1.21×** | **1.11×** |
| 3 | 2.67 | 1.17× | 1.08× |
| 4 | 2.94 | 1.09× | 1.01× |
| 7 | 3.24 | 0.82× | 0.75× |
| auto | 2.32 | 1.07× | 1.08× |

- Acceptance lengths are stable and hardware-independent; tok/s ratios are noisy (±0.1×).
- cap≈2–3 is the Apple Silicon operating point; the full block-7 verify is a net LOSS.

Qwen3-14B-4bit target (drafter passed explicitly — not in the port's registry):
baseline 61.7 tok/s; cap=1 1.05×; **cap=2 1.16× (accept 2.22)**; cap=3 1.03×; cap=7 0.77×.
The ratio does NOT improve much over 8B: the 14B drafter is proportionally bigger
(hidden 5120), so drafter cost tracks target cost. Expect ~1.1–1.2× from a straight port;
the Swift session must win on per-round overhead (fused single-sync rounds) and the
optimization levers below to beat it.

## Swift stack facts

1. **Debug builds halve fast-model decode.** Qwen3-8B-4bit: 47.9 tok/s (Debug) vs
   **103.6 tok/s (Release)** — matches the Python baseline (106.6) within 3%. The 27B
   BASELINE numbers were GPU-bound enough to hide `-Onone` overhead; 8B is not.
   **All DSpark benchmarks must use `-configuration Release`.**

2. **Verify-cost curve** (`mlzx-serve --bench-verify-curve`, Release, argmax consumed,
   median of 10):

   | ctx | kvBits=4 M=1 | M=2 | M=3 | M=8 | fp16 M=1 | fp16 M=8 |
   |-----|------|------|------|------|------|------|
   | 512 | 11.1ms | 17.1 | 18.2 | 33.0 | 10.1 | 35.9 |
   | 8192 | 12.9ms | **49.7** | 51.4 | 69.6 | 12.6 | 51.0 |
   | 32768 | 19.6ms | **137.4** | 138.7 | 186.3 | 19.6 | 86.7 |

   **Quantized-KV M>1 cliff**: with kvBits=4, the M=1 decode hits a fast qmv-path kernel;
   M≥2 falls onto a small-M `qmm` path that costs ~4–7× at ctx≥8k. fp16 KV has no cliff.
   This is MLX kernel dispatch, not fork routing (`attentionWithCacheUpdate` always uses
   the quantized path). Affects the existing MTP path too (verifies 2 tokens/step).
   Mitigation candidates (M4): per-row split of the verify attention into M independent
   M=1 qmv calls (SDPA rows are independent); or fp16-KV mode for DSpark; or bound
   confidence cap by context length.

3. **Ceiling projection** (ctx 512, kvBits=4, Qwen3-8B-4bit, drafter cost D≈4ms from the
   Python port's round timing): cap=2 → 2.29·11.07/(D+18.17) ≈ **1.14×**;
   cap=3 → ≈ **1.17×**. Consistent with the oracle's measured 1.11–1.21×.

## Gate decision: PASS (proceed to M1), eyes open

Real, reproducible gain on 8B but modest (~1.1–1.2×); the useful operating point is
cap 2–3, NOT the checkpoint's block 7. Upside levers beyond the base ratio: 14B target
(slower baseline), hybrid n-gram lookup drafting (the port commits ~6 tok/round on copy
runs), fixing the quantized-KV small-M verify path, fused single-sync greedy rounds.
Long-context DSpark with kvBits=4 is currently unattractive until the M>1 cliff is
addressed — the session should bound draft length (or disable drafting) at large offsets
until then.

## Follow-up #2 — adaptive draft on/off controller (2026-07-07, fork f7cfb21)

Sessions measure ms/committed-token for both arms (spec rounds vs pipelined plain-step
batches — lazily chained, one sync per batch, so the plain estimate matches the true
plain path) and run the faster one. Median windows (robust to per-request warmup
outliers), per-sample decisions, asymmetric margins (suspend only >8% loss; ties draft),
probe backoff, ONE controller shared per model. Kill switch MLXZ_DSPARK_ADAPTIVE=0.

Measured on the 8B default config: pathological losses clamp (sustained chat 0.876 →
0.94); bursty wins hold (math 1.08–1.20×, code 1.02–1.13×); sustained-SATURATED
converges to ~0.94–0.96 — hot ALU-bound spec rounds are genuinely ~5% slower than
plain, inside the tie band by design. Three controller designs (EWMA, symmetric-
hysteresis median, asymmetric median) all converge there: it is a hardware property,
not a tuning gap. mlxz's real workload is bursty/interactive, where drafting wins.
Lossless gate with the controller: 2.50%/token, within budget.
