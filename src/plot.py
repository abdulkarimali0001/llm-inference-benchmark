"""Charts from results/results.json -> results/quantization.png, results/context.png"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent / "results"


def main():
    d = json.loads((R / "results.json").read_text())
    q, c, bw = d["quantization"], d["context"], d.get("chip_bandwidth_gb_s")

    labels = [r["quant"] or r["model"].split("-")[-1] for r in q]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].bar(labels, [r["ollama_memory_gb"] or 0 for r in q], color="tab:gray"); ax[0].set(title="Model memory (GB)")
    ax[1].bar(labels, [r["decode_tok_s"] for r in q], color="tab:blue"); ax[1].set(title="Decode speed (tokens/s)")
    ax[2].bar(labels, [r.get("effective_bandwidth_gb_s") or 0 for r in q], color="tab:green")
    ax[2].set(title="Effective memory bandwidth (GB/s)")
    if bw:
        ax[2].axhline(bw, color="tab:red", ls="--", lw=1); ax[2].text(-0.4, bw * 0.92, f"chip spec {bw:g} GB/s", color="tab:red", fontsize=8)
    for a in ax:
        a.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Quantization: smaller weights = less memory to read per token ({q[0]['params']} model)", fontsize=11)
    fig.tight_layout(); fig.savefig(R / "quantization.png", dpi=150); plt.close(fig)

    x = [r["num_ctx"] for r in c]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(x, [r["prefill_tok_s"] for r in c], marker="o"); ax[0].set(title="Prefill speed (prompt tokens/s)")
    ax[1].plot(x, [r["decode_tok_s"] for r in c], marker="o", color="tab:orange"); ax[1].set(title="Decode speed (tokens/s)")
    ax[2].plot(x, [r["kv_cache_gb_theory"] or 0 for r in c], marker="o", color="tab:purple", label="KV cache (theory)")
    ax[2].plot(x, [r["ollama_memory_gb"] or 0 for r in c], marker="s", color="tab:gray", label="total reported by Ollama")
    ax[2].set(title="Memory (GB)"); ax[2].legend(fontsize=8)
    ax[1].set_ylim(0, max(r["decode_tok_s"] for r in c) * 1.2)  # start at 0 so run-to-run noise isn't exaggerated
    for a in ax:
        a.set_xscale("log", base=2); a.set_xticks(x, [f"{v:,}" for v in x]); a.minorticks_off()
        a.set_xlabel("context length (tokens)"); a.grid(alpha=0.3)
    fig.suptitle(f"Longer context: KV cache grows linearly, prefill slows, decode stays nearly flat ({c[0]['model']})", fontsize=11)
    fig.tight_layout(); fig.savefig(R / "context.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
