import json, sys, urllib.request

out = sys.argv[1]
prompts = [
    "What are the main tradeoffs between renting and buying a home? Keep it under 150 words.",
    "Write a Python function that merges two sorted lists without using sort().",
    "A recipe for 6 people needs 450 g flour. How much for 10 people?",
    "Write a six-line poem about a lighthouse keeper who is afraid of the dark.",
]
with open(out, "w") as f:
    for p in prompts:
        for _ in range(3):
            body = json.dumps({
                "model": "m",
                "messages": [{"role": "user", "content": p}],
                "temperature": 0.7, "max_tokens": 300,
            }).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:8199/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            m = r["choices"][0]["message"]
            f.write(json.dumps({
                "prompt": p,
                "reasoning": m.get("reasoning_content") or "",
                "content": m.get("content") or "",
            }) + "\n")
print("wrote", out)
