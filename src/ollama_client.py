"""Minimal client for a local Ollama server (https://ollama.com). No API key needed.

Set OLLAMA_HOST to change the address (default http://localhost:11434).
Only uses the Python standard library, so there is nothing extra to install.
"""
import json
import os
import urllib.error
import urllib.request

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


class OllamaError(RuntimeError):
    pass


def _call(path, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(HOST + path, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.URLError as e:
        raise OllamaError(f"Cannot reach Ollama at {HOST}. Is the Ollama app running? ({e})") from e


def generate(model, prompt, system=None, options=None, keep_alive="10m"):
    """Non-streaming completion. Returns Ollama's full JSON (response text + timing fields in ns)."""
    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive, "options": options or {}}
    if system:
        body["system"] = system
    return _call("/api/generate", body)


def chat(model, messages, options=None, fmt=None, keep_alive="10m"):
    body = {"model": model, "messages": messages, "stream": False, "keep_alive": keep_alive, "options": options or {}}
    if fmt:
        body["format"] = fmt  # e.g. "json"
    return _call("/api/chat", body)


def embed(model, texts):
    return _call("/api/embed", {"model": model, "input": texts})["embeddings"]


def show(model):
    return _call("/api/show", {"model": model})


def running():
    return _call("/api/ps").get("models", [])


def installed():
    return {m["name"] for m in _call("/api/tags").get("models", [])}


def pull(model):
    return _call("/api/pull", {"model": model, "stream": False}, timeout=3600)


def ensure(models):
    """Pull any model that isn't installed yet."""
    have = installed()
    for m in models:
        if m not in have and f"{m}:latest" not in have:
            print(f"pulling {m} (first time only)...", flush=True)
            pull(m)


def unload_all():
    """Free memory: ask Ollama to unload every model it has loaded (keep_alive=0)."""
    for m in running():
        _call("/api/generate", {"model": m["name"], "keep_alive": 0})
