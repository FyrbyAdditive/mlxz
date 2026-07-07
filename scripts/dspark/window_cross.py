import json, sys, urllib.request
body = json.dumps({"model": "m", "messages": [{"role": "user", "content":
    "Write a very long detailed essay (at least 1200 words) about the history of navigation at sea, covering ancient, medieval, and modern eras."}],
    "temperature": 0, "max_tokens": 1600}).encode()
req = urllib.request.Request("http://127.0.0.1:8199/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=1800))
m = r["choices"][0]["message"]
text = (m.get("reasoning_content") or "") + "␟" + (m.get("content") or "")
open(sys.argv[1], "w").write(text)
print(sys.argv[1], "tokens:", r["usage"]["completion_tokens"], "chars:", len(text))
