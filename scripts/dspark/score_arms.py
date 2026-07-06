"""Teacher-force each completion under the target model; compare the two arms' mean
per-token logprob with bootstrap 95% CIs. Overlapping CIs => no distributional drift
between speculative and plain sampling at temperature 0.7."""
import json, random
import mlx.core as mx
from mlx_lm import load

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")

def score(path):
    means = []
    for line in open(path):
        r = json.loads(line)
        completion = r["reasoning"] + r["content"]
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], add_generation_prompt=True)
        # The template pre-opens <think>; the emitted reasoning starts with "<think>\n"
        # again in the wire format — strip the duplicated opening tag if present.
        if completion.startswith("<think>"):
            completion = completion[len("<think>"):]
        comp_ids = tokenizer.encode(completion, add_special_tokens=False)
        if len(comp_ids) < 10:
            continue
        ids = prompt_ids + comp_ids
        logits = model(mx.array([ids]))[0]
        x = logits.astype(mx.float32); lp = x - mx.logsumexp(x, axis=-1, keepdims=True)
        n0 = len(prompt_ids)
        tok_lps = [lp[n0 - 1 + i, comp_ids[i]].item() for i in range(len(comp_ids))]
        means.append(sum(tok_lps) / len(tok_lps))
    return means

def boot_ci(xs, iters=10000):
    random.seed(7)
    ms = sorted(
        sum(random.choices(xs, k=len(xs))) / len(xs) for _ in range(iters))
    return ms[int(0.025 * iters)], ms[int(0.975 * iters)]

spec = score("arm-spec.jsonl")
plain = score("arm-plain.jsonl")
for name, xs in (("spec", spec), ("plain", plain)):
    lo, hi = boot_ci(xs)
    print(f"{name}: n={len(xs)} mean logprob/token = {sum(xs)/len(xs):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
