"""End-to-end test against a fake Ollama server (no real model needed).  Run: pytest -q"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import mock_ollama  # noqa: E402

srv, url = mock_ollama.start()
os.environ["OLLAMA_HOST"] = url

import bench  # noqa: E402


def test_kv_cache_formula():
    a = {"layers": 28, "kv_heads": 2, "head_dim": 128}
    # 2 * 28 * 2 * 128 * 4096 * 2 bytes = 117.4 MB
    assert abs(bench.kv_cache_gb(a, 4096) - 0.1174) < 1e-3


def test_full_run(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "R", tmp_path)
    import plot
    monkeypatch.setattr(plot, "R", tmp_path)
    monkeypatch.setattr(sys, "argv", ["bench", "--models", "mock", "--context-model", "mock",
                                      "--contexts", "512", "1024", "--repeats", "1", "--bandwidth", "100"])
    bench.main()
    for f in ["quantization_sweep.csv", "context_sweep.csv", "results.json", "quantization.png", "context.png"]:
        assert (tmp_path / f).exists()
