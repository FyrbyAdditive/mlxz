"""Pure mlx_lm, 4-bit weights + 4-bit QUANTIZED KV cache: greedy M=1 decode, then replay
the same tokens through 4-token forwards and count argmax flips. Bounds the divergence
ceiling for the mlxz default config (kvBits=4) with zero DSpark code involved."""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import QuantizedKVCache

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
prompts = [
    "Explain to a 10-year-old why the sky is blue and sunsets are red.",
    "Solve for x: 3(x - 4) + 7 = 2x + 11. Show each step.",
    "Write a Python function that merges two sorted lists into one sorted list without using sort().",
]
def qcache():
    return [QuantizedKVCache(group_size=64, bits=4) for _ in model.layers]

checked = flips = 0
for prompt in prompts:
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    cache = qcache()
    logits = model(mx.array([ids]), cache=cache)[0, -1]
    out = []
    for _ in range(200):
        tok = mx.argmax(logits).item()
        out.append(tok)
        logits = model(mx.array([[tok]]), cache=cache)[0, -1]
    cache2 = qcache()
    _ = model(mx.array([ids[:-1]]), cache=cache2)
    seq = [ids[-1]] + out
    for t in range(0, len(out) - 3, 4):
        lg = model(mx.array([seq[t : t + 4]]), cache=cache2)[0]
        pred = mx.argmax(lg, axis=-1).tolist()
        for j in range(4):
            checked += 1
            if pred[j] != out[t + j]:
                flips += 1
print(f"quantized-KV: checked={checked} flips={flips} ({100*flips/checked:.2f}%/token)")
