"""A fake Ollama server for tests, so the pipelines can be checked without real models.

It mimics the JSON shape of /api/generate, /api/chat, /api/embed, /api/show, /api/ps,
/api/tags and /api/pull. Answers are canned; timings scale with prompt length.
"""
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def fake_embedding(text, dim=64):
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=dim)
    # share direction for texts that share words, so retrieval is meaningful in tests
    for w in set(text.lower().split()):
        v += 0.6 * np.random.default_rng(int(hashlib.md5(w.encode()).hexdigest()[:8], 16)).normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


def answer_for(prompt):
    if "JSON" in prompt and "label" in prompt:
        return json.dumps({"label": "rejected", "reason": "mock judge"})
    if "JSON" in prompt and "question" in prompt:
        return json.dumps({"question": "HBM은 무엇의 약자인가요?", "answer": "High Bandwidth Memory"})
    if "verdict" in prompt:
        return json.dumps({"verdict": "correct"})
    return "서울입니다. 그런 것은 존재하지 않습니다. [1]"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/ps":
            self._send({"models": [{"name": "mock:latest", "size": 2_000_000_000, "size_vram": 2_000_000_000}]})
        elif self.path == "/api/tags":
            self._send({"models": [{"name": "mock:latest"}]})
        else:
            self._send({})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path == "/api/embed":
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return self._send({"embeddings": [fake_embedding(t) for t in texts]})
        if self.path == "/api/show":
            return self._send({"details": {"parameter_size": "1.5B", "quantization_level": "Q4_K_M", "family": "qwen2"},
                               "model_info": {"general.architecture": "qwen2", "qwen2.block_count": 28,
                                              "qwen2.attention.head_count": 12, "qwen2.attention.head_count_kv": 2,
                                              "qwen2.embedding_length": 1536, "qwen2.context_length": 32768}})
        if self.path == "/api/pull":
            return self._send({"status": "success"})
        prompt = body.get("prompt") or " ".join(m["content"] for m in body.get("messages", []))
        n_prompt = max(1, len(prompt) // 4)
        n_out = int(body.get("options", {}).get("num_predict", 32))
        text = answer_for(prompt)
        timing = {"total_duration": 10**9, "load_duration": 10**7, "prompt_eval_count": n_prompt,
                  "prompt_eval_duration": n_prompt * 200_000, "eval_count": n_out, "eval_duration": n_out * 20_000_000, "done": True}
        if self.path == "/api/chat":
            return self._send({"message": {"role": "assistant", "content": text}, **timing})
        return self._send({"response": text, **timing})


def start(port=0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
