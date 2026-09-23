"""Phase 6 step 4: tokens/sec with vs. without speculative decoding, same
warmup-then-measure, real-chat-prompt methodology as Phase 5B (a subset of its same 18
prompts, for continuity, rather than a fresh ad hoc set) -- not the full 18 here, since
both models resident simultaneously leaves very little VRAM headroom (peak ~9.9GB of
~9.86GB free measured in the correctness check), so this keeps the run comfortably
inside budget.
"""

import gc
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _PROJECT_ROOT)

from python.benchmarks.phase5b_end_to_end import PROMPTS as ALL_PROMPTS  # noqa: E402
from python.speculative.spec_decode import speculative_generate  # noqa: E402

TARGET_DIR = os.path.join(_PROJECT_ROOT, "..", "Evol_inference", "models", "phi-3-mini-4k-instruct")
DRAFT_DIR = os.path.join(_PROJECT_ROOT, "models", "tinyllama-1.1b-chat")
OUT_DIR = os.path.join(_PROJECT_ROOT, "build", "phase6")

MAX_NEW_TOKENS = 64
K = 4
WARMUP_PROMPT = "Say hello in one short sentence."
BENCH_PROMPT_IDS = ["s1", "s2", "m1", "m2", "l1", "l2"]  # 2 short / 2 medium / 2 long


def timed_plain(model, tokenizer, prompt_text, max_new_tokens):
    messages = [{"role": "user", "content": prompt_text}]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model.generate(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def timed_speculative(target, draft, tokenizer, prompt_text, max_new_tokens, k):
    messages = [{"role": "user", "content": prompt_text}]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(target.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    _out, stats = speculative_generate(target, draft, enc.input_ids, max_new_tokens=max_new_tokens, k=k)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000.0
    return ms, stats


def main():
    print(f"Loading target (Phi-3-mini) and draft (TinyLlama-1.1B) models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_DIR)
    target = AutoModelForCausalLM.from_pretrained(TARGET_DIR, dtype=torch.float16, device_map="cuda").eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_DIR, dtype=torch.float16, device_map="cuda").eval()

    prompts = [p for p in ALL_PROMPTS if p["id"] in BENCH_PROMPT_IDS]
    assert len(prompts) == len(BENCH_PROMPT_IDS)

    print("warmup (plain)...")
    timed_plain(target, tokenizer, WARMUP_PROMPT, MAX_NEW_TOKENS)
    print("warmup (speculative)...")
    timed_speculative(target, draft, tokenizer, WARMUP_PROMPT, MAX_NEW_TOKENS, K)

    results = []
    for p in prompts:
        plain_ms = timed_plain(target, tokenizer, p["text"], MAX_NEW_TOKENS)
        plain_tok_s = MAX_NEW_TOKENS / (plain_ms / 1000.0)

        spec_ms, stats = timed_speculative(target, draft, tokenizer, p["text"], MAX_NEW_TOKENS, K)
        spec_tok_s = stats.tokens_generated / (spec_ms / 1000.0)

        row = {
            "id": p["id"], "category": p["category"],
            "plain_ms": plain_ms, "plain_tok_s": plain_tok_s,
            "spec_ms": spec_ms, "spec_tok_s": spec_tok_s,
            "speedup": spec_tok_s / plain_tok_s,
            "acceptance_rate": stats.acceptance_rate,
            "rounds": stats.rounds,
            "tokens_generated": stats.tokens_generated,
        }
        results.append(row)
        print(f"  [{p['id']:4s} {p['category']:6s}] plain {plain_tok_s:6.1f} tok/s  "
              f"spec {spec_tok_s:6.1f} tok/s  ({row['speedup']:.2f}x)  "
              f"acceptance={stats.acceptance_rate:.2f}  rounds={stats.rounds}")

    mean_speedup = statistics.mean(r["speedup"] for r in results)
    mean_acceptance = statistics.mean(r["acceptance_rate"] for r in results)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "max_new_tokens": MAX_NEW_TOKENS, "k": K,
            "mean_speedup": mean_speedup, "mean_acceptance_rate": mean_acceptance,
            "per_prompt": results,
        }, f, indent=2)

    print(f"\n=== Summary (n={len(results)} prompts, k={K}, {MAX_NEW_TOKENS} tokens each) ===")
    print(f"Mean tok/s speedup (speculative vs. plain): {mean_speedup:.2f}x")
    print(f"Mean draft acceptance rate: {mean_acceptance:.2f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
