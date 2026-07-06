# DSpark M0 — go/no-go findings (M5 Max, 2026-07-06/07)

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
