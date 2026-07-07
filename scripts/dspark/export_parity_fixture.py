#!/usr/bin/env python3
"""Export DSpark drafter parity fixtures for the Swift port's M1 gate.

Runs the reference implementation (mlx-dspark, MIT) drafter — bf16, unquantized — on
deterministic synthetic target hidden states and records inputs + outputs. The Swift test
(DSparkDrafterTests.testCheckpointParity) replays the same inputs through the Swift port
of the same checkpoint and compares logits/tokens/confidence.

Two rounds are exported: round 1 drafts a block after a 48-position context; round 2
appends 3 more context positions (as after a verify commit) and drafts again — catching
rope/offset bookkeeping bugs a single round can't.

Usage:
  python export_parity_fixture.py deepseek-ai/dspark_qwen3_8b_block7 fixture.safetensors
"""

import sys

import mlx.core as mx
from mlx_dspark.load import load_drafter


def main() -> None:
    repo, out = sys.argv[1], sys.argv[2]
    drafter, cfg = load_drafter(repo, quantize=False)

    mx.random.seed(42)
    t1, t2 = 48, 3
    n_tap = len(cfg.target_layer_ids)
    # bf16 like real taps; saved as f32 (exact) — the Swift side casts back to bf16.
    ctx1 = mx.random.normal((1, t1, n_tap * cfg.hidden_size)).astype(mx.bfloat16)
    ctx2 = mx.random.normal((1, t2, n_tap * cfg.hidden_size)).astype(mx.bfloat16)

    caches = drafter.make_ctx_cache()
    pending = 9906
    k = cfg.block_size
    tensors = {
        "ctx1": ctx1.astype(mx.float32),
        "ctx2": ctx2.astype(mx.float32),
        "meta": mx.array([pending, cfg.mask_token_id, t1, t2], dtype=mx.int32),
    }

    drafter.update_context(ctx1, ctx_offset=0, ctx_caches=caches)
    for round_idx, offset in ((1, t1), (2, t1 + t2)):
        if round_idx == 2:
            drafter.update_context(ctx2, ctx_offset=t1, ctx_caches=caches)
        block_ids = [pending] + [cfg.mask_token_id] * (k - 1)
        noise = drafter.embed(mx.array([block_ids]))
        if round_idx == 1:
            fused = drafter.fuse_target(ctx1)
            layer0 = drafter.layers[0](noise, offset, caches[0])
            mx.eval(fused, layer0, noise)
            tensors["r1_fused"] = fused.astype(mx.float32)
            tensors["r1_layer0"] = layer0[0].astype(mx.float32)
            tensors["r1_noise"] = noise[0].astype(mx.float32)
        block_hidden = drafter.backbone(noise, offset, caches)
        base_logits = drafter.compute_logits(block_hidden)[0]  # [k, V]
        draft = drafter.sample_block(base_logits, first_prev_token=pending)
        prev = mx.concatenate([mx.array([pending]), draft[:-1]])
        conf = drafter.confidence_logits(block_hidden[0], prev)
        mx.eval(block_hidden, base_logits, draft, conf)
        tensors[f"r{round_idx}_block_hidden"] = block_hidden[0].astype(mx.float32)
        tensors[f"r{round_idx}_base_logits"] = base_logits.astype(mx.float32)
        tensors[f"r{round_idx}_draft"] = draft.astype(mx.int32)
        tensors[f"r{round_idx}_conf"] = conf.astype(mx.float32)
        print(f"round {round_idx}: draft={draft.tolist()}")

    mx.save_safetensors(out, tensors)
    print("wrote", out)


if __name__ == "__main__":
    main()
