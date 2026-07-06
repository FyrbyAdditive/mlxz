"""At every greedy decode position, measure the top-2 logit gap of the target model.
If a nontrivial fraction of positions have gaps below GPU-kernel noise (~1e-3..1e-2),
then any M=1-vs-M>1 kernel difference must flip greedy output at that rate — the
mechanism behind spec-vs-plain divergence, independent of any implementation bug."""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
prompts = [
    "Explain to a 10-year-old why the sky is blue and sunsets are red.",
    "Write a SQL query that finds the three highest-spending customers per region from tables orders(customer_id, region, amount) and customers(id, name).",
    "Solve for x: 3(x - 4) + 7 = 2x + 11. Show each step.",
]
buckets = {1e-4: 0, 1e-3: 0, 1e-2: 0, 1e-1: 0}
total = 0
for prompt in prompts:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True)
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)[0, -1]
    for _ in range(256):
        pair = mx.sort(logits.astype(mx.float32))[-2:]
        gap = (pair[1] - pair[0]).item()
        tok = mx.argmax(logits).item()
        total += 1
        for b in buckets:
            if gap < b: buckets[b] += 1
        if tok in (tokenizer.eos_token_id,): break
        logits = model(mx.array([[tok]]), cache=cache)[0, -1]
print(f"positions={total}")
for b in sorted(buckets):
    print(f"  top-2 gap < {b:g}: {buckets[b]} ({100*buckets[b]/total:.2f}%)")
