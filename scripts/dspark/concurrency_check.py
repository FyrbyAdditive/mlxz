#!/usr/bin/env python3
"""Concurrency check: N simultaneous greedy requests against a running mlxz-serve.
Reports per-request wall/tokens and the AGGREGATE tokens/sec. Run with DSpark on and
off; DSpark must not make the c=N aggregate worse than plain (rejected-draft work is
wasted compute under contention — the fair scheduler should keep it bounded).

Usage: concurrency_check.py [port] [n]
"""
import json
import sys
import threading
import time
import urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8199"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
url = f"http://127.0.0.1:{port}/v1/chat/completions"

prompts = [
    "Explain how a heat pump works in one paragraph.",
    "Write a Python function to reverse words in a sentence.",
    "What is 17% of 2,340? Show the working.",
    "Give three tips for photographing the night sky.",
][:n]

results = [None] * len(prompts)

def worker(i: int, prompt: str) -> None:
    body = json.dumps({
        "model": "m", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    results[i] = (time.time() - t0, r["usage"]["completion_tokens"])

t0 = time.time()
threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(prompts)]
for t in threads: t.start()
for t in threads: t.join()
wall = time.time() - t0

total_tokens = sum(tok for _, tok in results)
for i, (secs, tok) in enumerate(results):
    print(f"req {i}: {tok} tok in {secs:.2f}s ({tok/secs:.1f} tok/s)")
print(f"aggregate: {total_tokens} tok in {wall:.2f}s = {total_tokens/wall:.1f} tok/s (c={len(prompts)})")
