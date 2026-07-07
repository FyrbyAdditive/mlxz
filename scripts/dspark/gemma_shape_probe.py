"""Pure-Python ceiling for gemma4-unified-8bit: greedy M=1 decode, then replay the SAME
tokens through M=4 forwards; count argmax flips. No DSpark code."""
import mlx.core as mx
from mlx_vlm.utils import load_model
from mlx_dspark.target import Target
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from pathlib import Path
path = Path(snapshot_download("mlx-community/gemma-4-12B-it-8bit"))
model = load_model(path)
tokenizer = AutoTokenizer.from_pretrained(path)
target = Target(model, tokenizer)
prompts = [
    "Explain to a 10-year-old why the sky is blue and sunsets are red.",
    "Solve for x: 3(x - 4) + 7 = 2x + 11. Show each step.",
    "Write a Python function that merges two sorted lists without using sort().",
]
checked = flips = 0
for prompt in prompts:
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
    ids = list(ids) if not hasattr(ids, "input_ids") else list(ids.input_ids)
    cache = target.make_cache()
    logits = target.plain(mx.array([ids]), cache)[0, -1]
    out = []
    for _ in range(200):
        tok = mx.argmax(logits).item()
        out.append(tok)
        logits = target.plain(mx.array([[tok]]), cache)[0, -1]
    cache2 = target.make_cache()
    _ = target.plain(mx.array([ids[:-1]]), cache2)
    seq = [ids[-1]] + out
    for t in range(0, len(out) - 3, 4):
        lg = target.plain(mx.array([seq[t : t + 4]]), cache2)[0]
        pred = mx.argmax(lg, axis=-1).tolist()
        for j in range(4):
            checked += 1
            if pred[j] != out[t + j]:
                flips += 1
print(f"gemma4-8bit: checked={checked} flips={flips} ({100*flips/checked:.2f}%/token)")
