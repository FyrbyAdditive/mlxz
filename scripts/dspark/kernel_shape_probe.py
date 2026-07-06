"""Pure mlx_lm probe (no DSpark code anywhere): greedy-decode M=1, then replay the SAME
tokens through 4-token forwards (the verify shape) and count positions where argmax
differs. Any flips here are purely kernel-shape numerics — the ceiling for what ANY
speculative implementation can achieve on byte-identity."""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
prompt = "Explain to a 10-year-old why the sky is blue and sunsets are red."
ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)

# Arm A: plain greedy, one token per forward.
cache = make_prompt_cache(model)
logits = model(mx.array([ids]), cache=cache)[0, -1]
out = []
for _ in range(200):
    tok = mx.argmax(logits).item()
    out.append(tok)
    logits = model(mx.array([[tok]]), cache=cache)[0, -1]

# Arm B: teacher-force the SAME tokens in 4-token chunks; compare per-position argmax.
cache2 = make_prompt_cache(model)
_ = model(mx.array([ids[:-1]]), cache=cache2)
seq = [ids[-1]] + out  # inputs; model(seq[t]) predicts out[t]
flips = 0
checked = 0
for t in range(0, len(out) - 3, 4):
    chunk = mx.array([seq[t : t + 4]])
    lg = model(chunk, cache=cache2)[0]
    pred = mx.argmax(lg, axis=-1).tolist()
    for j in range(4):
        checked += 1
        if pred[j] != out[t + j]:
            flips += 1
print(f"checked={checked} flips={flips} ({100*flips/checked:.2f}%/token)")
