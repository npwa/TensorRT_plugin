"""Phase 6: correctness check for speculative_generate -- its output is mathematically
required to be byte-identical to plain greedy `target_model.generate(do_sample=False)`
on the same prompt (see spec_decode.py's docstring for why). This script proves that
holds before any benchmark number from it is trusted, the same "verify, don't assume"
discipline the rest of this project has applied at every phase.
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _PROJECT_ROOT)

from python.speculative.spec_decode import speculative_generate  # noqa: E402

TARGET_DIR = os.path.join(_PROJECT_ROOT, "..", "Evol_inference", "models", "phi-3-mini-4k-instruct")
DRAFT_DIR = os.path.join(_PROJECT_ROOT, "models", "tinyllama-1.1b-chat")

PROMPTS = [
    "What is the capital of France?",
    "Write a short poem about the changing seasons.",
    "Explain how a neural network learns, in two paragraphs.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    print("Loading target (Phi-3-mini) and draft (TinyLlama-1.1B) models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_DIR)
    target = AutoModelForCausalLM.from_pretrained(TARGET_DIR, dtype=torch.float16, device_map="cuda").eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_DIR, dtype=torch.float16, device_map="cuda").eval()

    all_match = True
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(target.device)

        with torch.no_grad():
            plain_out = target.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_new_tokens, min_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        plain_new = plain_out[0, enc.input_ids.shape[1]:].tolist()

        spec_out, stats = speculative_generate(
            target, draft, enc.input_ids, max_new_tokens=args.max_new_tokens, k=args.k,
        )
        spec_new = spec_out[0, enc.input_ids.shape[1]:].tolist()

        match = plain_new == spec_new
        all_match &= match
        print(f"\nPrompt: {prompt!r}")
        print(f"  match: {match}  (rounds={stats.rounds}, acceptance_rate={stats.acceptance_rate:.2f}, "
              f"accepted/round={stats.accepted_per_round})")
        if not match:
            print(f"  plain: {tokenizer.decode(plain_new)!r}")
            print(f"  spec : {tokenizer.decode(spec_new)!r}")

    print(f"\n{'ALL PROMPTS MATCH' if all_match else 'MISMATCH FOUND'}")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
