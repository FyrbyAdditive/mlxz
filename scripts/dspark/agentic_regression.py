#!/usr/bin/env python3
"""Agentic prefix-reuse regression: 3-turn conversation with a shared ~2k-token system
prompt against a running mlxz-serve (port from argv[1], default 8199). Turn 2/3 must
reuse the prefix snapshot (server logs [PREFIX] reused=N with MLXZ_PREFIX_DIAG=1) and
their TTFT must be far below turn 1's. Run once with DSpark on and once with
--dspark-draft off; the reuse behavior and TTFT must match within noise.

Prints per-turn TTFT (client-side, streaming) and total time.
"""
import json
import sys
import time
import urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8199"
url = f"http://127.0.0.1:{port}/v1/chat/completions"

system = ("You are a meticulous engineering assistant for the ACME robotics stack. "
          + "Rules: " + " ".join(f"Rule {i}: always consider constraint {i} about "
          "thermal, torque, latency, and safety margins in subsystem design." for i in range(120)))
turns = [
    "Summarize your rules in one sentence.",
    "Now list the three most important constraint categories.",
    "Which rule number covers latency? Answer briefly.",
]

messages = [{"role": "system", "content": system}]
for t, user in enumerate(turns, 1):
    messages.append({"role": "user", "content": user})
    body = json.dumps({
        "model": "m", "messages": messages, "temperature": 0,
        "max_tokens": 120, "stream": True,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    content = ""
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            delta = chunk["choices"][0]["delta"]
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece and ttft is None:
                ttft = time.time() - t0
            content += piece
    total = time.time() - t0
    print(f"turn {t}: ttft={ttft:.2f}s total={total:.2f}s chars={len(content)}")
    messages.append({"role": "assistant", "content": content})
