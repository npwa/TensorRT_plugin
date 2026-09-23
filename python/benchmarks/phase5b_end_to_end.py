"""Phase 5B: does the kernel's op-level speedup survive a full, real, end-to-end chat
response?

Phase 5's benchmark table measured one linear layer in isolation (one shape, batch of
2) and found the optimized CUDA kernel ~9x faster than plain PyTorch eager. This script
tests whether that holds up once the kernel is one piece of a real, full Phi-3-mini
generate() call -- all 32 transformer blocks, real tokenizer, real chat prompts of
varying complexity -- rather than leaving it an open caveat.

Two configurations share IDENTICAL quantized weights (same `quantize_groupwise_int4`
call per layer) and differ only in which code computes the matmul:
  (b) EagerInt4Linear  -- plain PyTorch dequant-then-matmul (today's slow path)
  (c) KernelInt4Linear -- the Phase 2 optimized fused CUDA kernel
An FP16 baseline is included for context, not as the primary comparison.

Usage:
  .venv/bin/python python/benchmarks/phase5b_end_to_end.py [--smoke] [--max-new-tokens N]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time

import torch
from scipy import stats as sp_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _PROJECT_ROOT)
os.environ["PATH"] = os.path.join(_PROJECT_ROOT, ".venv", "bin") + os.pathsep + os.environ.get("PATH", "")

from python.benchmarks.phase5b_modules import EagerInt4Linear, KernelInt4Linear, restore_all, substitute_all  # noqa: E402

MODEL_DIR = os.path.join(_PROJECT_ROOT, "..", "Evol_inference", "models", "phi-3-mini-4k-instruct")
OUT_DIR = os.path.join(_PROJECT_ROOT, "build", "phase5b")

WARMUP_PROMPT = "Say hello in one short sentence."

# 18 real chat-style prompts spanning complexity -- 6 short/simple, 6 medium, 6
# long/multi-part -- so "complexity" varies mainly through prompt (prefill) length and
# content while decode length stays fixed and controlled (see --max-new-tokens).
PROMPTS = [
    {"id": "s1", "category": "short", "text": "What is the capital of France?"},
    {"id": "s2", "category": "short", "text": "Name three colors of the rainbow."},
    {"id": "s3", "category": "short", "text": "What year did World War II end?"},
    {"id": "s4", "category": "short", "text": "Convert 10 miles to kilometers."},
    {"id": "s5", "category": "short", "text": "What's the chemical symbol for gold?"},
    {"id": "s6", "category": "short", "text": "Give me a synonym for 'happy'."},
    {"id": "m1", "category": "medium", "text": "Explain the difference between a stack and a queue in simple terms."},
    {"id": "m2", "category": "medium", "text": "Write a short poem about the changing seasons."},
    {"id": "m3", "category": "medium", "text": "Summarize why photosynthesis is important for life on Earth."},
    {"id": "m4", "category": "medium", "text": "What are three pros and three cons of remote work?"},
    {"id": "m5", "category": "medium", "text": "Explain how a neural network learns, in two paragraphs."},
    {"id": "m6", "category": "medium", "text": "Write a Python function that checks if a number is prime."},
    {"id": "l1", "category": "long", "text": (
        "I'm planning a 5-day trip to Japan focused on food and history. Suggest a rough "
        "day-by-day itinerary, including one city per day and one must-try dish for each."
    )},
    {"id": "l2", "category": "long", "text": (
        "A train leaves city A at 60 mph heading toward city B, 300 miles away. Another "
        "train leaves city B at the same time heading toward city A at 40 mph. How long "
        "until they meet, and how far from city A? Show your reasoning step by step."
    )},
    {"id": "l3", "category": "long", "text": (
        "Write a short story about a robot who discovers music for the first time, "
        "including a moment of conflict and a resolution."
    )},
    {"id": "l4", "category": "long", "text": (
        "Compare and contrast supervised, unsupervised, and reinforcement learning, with "
        "one real-world example of each."
    )},
    {"id": "l5", "category": "long", "text": (
        "I have a dataset with missing values, outliers, and mixed categorical/numerical "
        "columns. Walk me through a reasonable data-cleaning plan, step by step, and "
        "explain why each step matters."
    )},
    {"id": "l6", "category": "long", "text": (
        "Explain the causes of the French Revolution, covering economic, social, and "
        "political factors, in a structured multi-paragraph answer."
    )},
]


@torch.no_grad()
def timed_generate(model, tokenizer, prompt_text: str, max_new_tokens: int) -> tuple[float, int]:
    """Wall-clock ms for one full generate() call, forced to produce exactly
    `max_new_tokens` tokens (min_new_tokens == max_new_tokens) so a comparison across
    configurations times equal decode work, not incidentally-different response lengths
    from an early EOS under one configuration but not another."""
    messages = [{"role": "user", "content": prompt_text}]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = enc.input_ids.shape[1]

    torch.cuda.synchronize()
    start = time.perf_counter()
    model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, prompt_len


def run_configuration(model, tokenizer, prompts, max_new_tokens, label):
    print(f"  warmup ({label})...", flush=True)
    timed_generate(model, tokenizer, WARMUP_PROMPT, max_new_tokens)

    results = []
    for p in prompts:
        ms, prompt_len = timed_generate(model, tokenizer, p["text"], max_new_tokens)
        print(f"    [{label}] {p['id']:4s} ({p['category']:6s}, prompt_len={prompt_len:3d}): {ms:8.1f} ms")
        results.append({**p, "prompt_len": prompt_len, "latency_ms": ms})
    return results


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    """Mean and 95% confidence interval half-width via the t-distribution (not a normal
    z=1.96 approximation) -- appropriate for a sample this small (n=18)."""
    n = len(values)
    m = statistics.mean(values)
    sem = statistics.stdev(values) / (n ** 0.5)
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    half_width = t_crit * sem
    return m, m - half_width, m + half_width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny run (2 prompts, 8 tokens) to validate the pipeline")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    prompts = PROMPTS[:2] if args.smoke else PROMPTS
    max_new_tokens = 8 if args.smoke else args.max_new_tokens

    print(f"Loading tokenizer + FP16 model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16, device_map="cuda")
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    print("\n=== (a) FP16 baseline ===")
    all_results["fp16"] = run_configuration(model, tokenizer, prompts, max_new_tokens, "fp16")

    print("\n=== (b) INT4 via plain PyTorch eager ===")
    originals = substitute_all(model, EagerInt4Linear)
    all_results["eager_int4"] = run_configuration(model, tokenizer, prompts, max_new_tokens, "eager_int4")
    restore_all(model, originals)
    del originals
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== (c) INT4 via the optimized CUDA kernel ===")
    originals = substitute_all(model, KernelInt4Linear)
    all_results["kernel_int4"] = run_configuration(model, tokenizer, prompts, max_new_tokens, "kernel_int4")
    restore_all(model, originals)
    del originals
    gc.collect()
    torch.cuda.empty_cache()

    # Per-prompt speedup: the effect Phase 5 measured at the op level, now measured
    # across a full generate() call -- eager_int4 vs kernel_int4, same weights.
    per_prompt = []
    for eager_r, kernel_r in zip(all_results["eager_int4"], all_results["kernel_int4"]):
        assert eager_r["id"] == kernel_r["id"]
        speedup = eager_r["latency_ms"] / kernel_r["latency_ms"]
        per_prompt.append({
            "id": eager_r["id"], "category": eager_r["category"], "prompt_len": eager_r["prompt_len"],
            "eager_ms": eager_r["latency_ms"], "kernel_ms": kernel_r["latency_ms"], "speedup": speedup,
        })

    speedups = [r["speedup"] for r in per_prompt]
    mean_speedup, ci_lo, ci_hi = mean_ci95(speedups)

    summary = {
        "max_new_tokens": max_new_tokens,
        "n_prompts": len(prompts),
        "per_prompt": per_prompt,
        "mean_speedup": mean_speedup,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "raw": all_results,
    }
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary (n={len(prompts)} prompts, {max_new_tokens} forced decode tokens each) ===")
    print(f"Mean end-to-end speedup (kernel vs eager, same weights): {mean_speedup:.2f}x  "
          f"(95% CI: {ci_lo:.2f}x - {ci_hi:.2f}x)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
