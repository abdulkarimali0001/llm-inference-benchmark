"""LLM inference benchmark on a laptop: how quantization and context length change
speed and memory, and why token generation is limited by memory bandwidth.

Usage (Ollama app running):
    python src/bench.py                      # default models for a MacBook with 8 GB+
    python src/bench.py --models qwen2.5:7b-instruct-q4_K_M --contexts 512 2048 8192
    python src/bench.py --bandwidth 100      # your chip's memory bandwidth in GB/s, for the comparison

Two experiments:
  A. Quantization sweep: the same model at FP16, Q8 and Q4, fixed prompt length.
  B. Context sweep: one model, prompts from 512 to 8,192 tokens.

For each run we record (median of --repeats runs):
  prefill speed (prompt tokens/s), decode speed (generated tokens/s), time to first token,
  model memory reported by Ollama, and the theoretical KV-cache size.

Why decode speed is about memory: to generate ONE token, the chip must read every weight
of the model from memory. So   decode tokens/s x model size (GB)  ~= effective memory
bandwidth (GB/s). This is exactly why AI accelerators need high-bandwidth memory (HBM).
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import ollama_client as ol

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
DEFAULT_QUANT = ["qwen2.5:1.5b-instruct-fp16", "qwen2.5:1.5b-instruct-q8_0", "qwen2.5:1.5b-instruct-q4_K_M"]
FILLER = ("Semiconductor memory stores data in cells. DRAM keeps each bit as charge in a capacitor that must be "
          "refreshed, while NAND flash traps charge in floating gates and keeps data without power. ")


def make_prompt(n_tokens, salt):
    """A prompt of roughly n_tokens tokens. The random salt at the start defeats Ollama's prompt cache,
    so every run really processes the whole prompt."""
    words_needed = int(n_tokens * 0.75)
    body = (FILLER * (words_needed // len(FILLER.split()) + 1)).split()[:words_needed]
    return f"[run {salt}] " + " ".join(body) + "\n\nSummarize the text above in one sentence."


def arch_info(model):
    """Layers, KV heads and head size from the model file, for the KV-cache formula."""
    info = ol.show(model)
    mi, det = info.get("model_info", {}), info.get("details", {})
    a = mi.get("general.architecture", "")
    g = lambda k: mi.get(f"{a}.{k}")
    heads, kv, emb = g("attention.head_count"), g("attention.head_count_kv"), g("embedding_length")
    return {"layers": g("block_count"), "heads": heads, "kv_heads": kv or heads,
            "head_dim": (emb // heads) if emb and heads else None,
            "params": det.get("parameter_size"), "quant": det.get("quantization_level")}


def kv_cache_gb(a, ctx, bytes_per_value=2):
    """K and V tensors for every layer: 2 x layers x kv_heads x head_dim x tokens x bytes (FP16 cache)."""
    if not all([a["layers"], a["kv_heads"], a["head_dim"]]):
        return None
    return 2 * a["layers"] * a["kv_heads"] * a["head_dim"] * ctx * bytes_per_value / 1e9


def model_mem_gb(model):
    for m in ol.running():
        if m["name"] in (model, f"{model}:latest") or m.get("model") == model:
            return m.get("size", 0) / 1e9
    return None


def run_once(model, n_prompt, n_out, ctx, salt):
    opts = {"num_ctx": ctx, "num_predict": n_out, "temperature": 0, "seed": 1}
    t0 = time.perf_counter()
    r = ol.generate(model, make_prompt(n_prompt, salt), options=opts)
    wall = time.perf_counter() - t0
    pe, pd = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 1) / 1e9
    ec, ed = r.get("eval_count", 0), r.get("eval_duration", 1) / 1e9
    return {"prompt_tokens": pe, "output_tokens": ec, "prefill_tok_s": pe / pd if pd else None,
            "decode_tok_s": ec / ed if ed else None, "ttft_s": r.get("load_duration", 0) / 1e9 + pd, "wall_s": wall}


def measure(model, n_prompt, n_out, ctx, repeats):
    # Unload other models first: on an 8 GB Mac several loaded models force swapping,
    # which slows everything down and makes the memory reading wrong.
    ol.unload_all()
    run_once(model, 64, 8, ctx, "warmup")  # load the model and allocate the context first
    runs = [run_once(model, n_prompt, n_out, ctx, random.randint(0, 10**9)) for _ in range(repeats)]
    med = {k: statistics.median([r[k] for r in runs if r[k] is not None]) for k in runs[0]}
    a = arch_info(model)
    mem = model_mem_gb(model)
    row = {"model": model, "params": a["params"], "quant": a["quant"], "num_ctx": ctx, **{k: round(v, 3) for k, v in med.items()},
           "ollama_memory_gb": round(mem, 3) if mem else None,
           "kv_cache_gb_theory": round(kv_cache_gb(a, ctx), 4) if kv_cache_gb(a, ctx) else None}
    # weights read per generated token ~ memory minus KV cache; use it to estimate effective bandwidth
    if mem and med["decode_tok_s"]:
        weights = mem - (row["kv_cache_gb_theory"] or 0)
        row["effective_bandwidth_gb_s"] = round(med["decode_tok_s"] * weights, 1)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.write_text(",".join(keys) + "\n" + "".join(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n" for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_QUANT, help="quantization sweep models (same base model)")
    ap.add_argument("--context-model", default="qwen2.5:1.5b-instruct-q4_K_M")
    ap.add_argument("--contexts", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--bandwidth", type=float, default=None, help="chip memory bandwidth in GB/s (for the chart)")
    args = ap.parse_args()
    R.mkdir(exist_ok=True)
    ol.ensure(set(args.models) | {args.context_model})

    print("== A. quantization sweep ==")
    quant = [measure(m, args.prompt_tokens, args.output_tokens, max(2048, args.prompt_tokens * 2), args.repeats) for m in args.models]
    print("== B. context sweep ==")
    ctx = [measure(args.context_model, int(c * 0.9), args.output_tokens, c, args.repeats) for c in args.contexts]

    write_csv(quant, R / "quantization_sweep.csv"); write_csv(ctx, R / "context_sweep.csv")
    (R / "results.json").write_text(json.dumps({"quantization": quant, "context": ctx, "chip_bandwidth_gb_s": args.bandwidth}, indent=2))
    from plot import main as plot
    plot()


if __name__ == "__main__":
    main()
