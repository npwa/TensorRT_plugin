"""Phase 6: greedy speculative decoding (requirements.md §8's selected stretch option).

Standard speculative decoding, restricted to the greedy/deterministic case (matching
this project's do_sample=False convention throughout, Phase 5B included): the draft
model proposes K tokens autoregressively; the target model verifies all K in a single
batched forward pass; a drafted token is accepted iff it equals the target's own greedy
choice at that position. This gives a hard, checkable correctness property --
`speculative_generate`'s output is mathematically required to be byte-identical to
plain `target_model.generate(..., do_sample=False)` on the same input, since every
emitted token is always the target's own greedy pick (either confirmed via the draft,
or substituted directly when the draft was wrong). The speedup comes entirely from
collapsing what would be K sequential target forward passes into one batched pass,
whenever the draft's guesses are right.

Both models' KV caches are carried across rounds (not recomputed from scratch each
round) -- necessary for the tokens/sec comparison against plain generation to be a fair,
apples-to-apples measurement rather than one artificially slowed by O(n^2) reprocessing.

Vocab mismatch note: the target's tokenizer (Phi-3) has a larger vocabulary than the
draft's (TinyLlama) -- Phi-3's chat template inserts special formatting tokens (e.g.
`<|user|>`, `<|end|>`) with ids >= the draft's vocab size, and the target's own greedy
choice can legitimately land on one of those ids too (its end-of-turn token, say). Any
such id fed to the draft model as an embedding lookup index would crash. This never
threatens correctness of the *output* -- the draft is only ever a source of guesses that
get independently checked against the target's real predictions -- so out-of-range ids
are simply clamped to 0 before being fed to the draft; a clamped id can only make the
draft's next guess worse (hurting the acceptance rate, i.e. speed), never wrong in a way
that changes what speculative_generate returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SpecDecodeStats:
    rounds: int = 0
    tokens_generated: int = 0
    tokens_proposed: int = 0  # sum of this_k across rounds (last round may be < k)
    tokens_accepted_from_draft: int = 0  # excludes the always-present correction/bonus token
    accepted_per_round: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of *drafted* tokens (not counting the guaranteed extra token each
        round contributes) that the target accepted."""
        return self.tokens_accepted_from_draft / max(1, self.tokens_proposed)


def _draft_safe(token_ids: torch.Tensor, draft_vocab_size: int) -> torch.Tensor:
    """Clamp any id the draft model's embedding table can't index (see module
    docstring's vocab mismatch note) to 0. Never applied to the draft's own outputs --
    only to tokens computed by/from the target that get fed *into* the draft."""
    return torch.where(token_ids < draft_vocab_size, token_ids, torch.zeros_like(token_ids))


@torch.no_grad()
def speculative_generate(
    target_model, draft_model, input_ids: torch.Tensor, max_new_tokens: int, k: int = 4,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, SpecDecodeStats]:
    """Greedy speculative decoding. `input_ids`: [1, prompt_len] on the target's device
    (both models must already be on the same device). Returns (full_sequence,
    stats) where full_sequence is [1, prompt_len + n] with n <= max_new_tokens (fewer
    only if `eos_token_id` is hit and stops generation early)."""
    stats = SpecDecodeStats()
    draft_vocab_size = draft_model.config.vocab_size

    target_out = target_model(input_ids=input_ids, use_cache=True)
    target_cache = target_out.past_key_values
    target_next_logits = target_out.logits[:, -1, :]

    draft_out = draft_model(input_ids=_draft_safe(input_ids, draft_vocab_size), use_cache=True)
    draft_cache = draft_out.past_key_values
    # Prediction for the first new token, carried forward exactly like
    # `target_next_logits` -- NOT re-derived by re-feeding `generated[:, -1:]` into the
    # draft, which would be the prompt's *last already-cached* token (and, in Phi-3's
    # case, frequently one of its own special ids the draft can't even embed).
    draft_next_logits = draft_out.logits[:, -1, :]

    generated = input_ids
    n_new = 0
    hit_eos = False

    while n_new < max_new_tokens and not hit_eos:
        remaining = max_new_tokens - n_new
        this_k = min(k, remaining)  # never draft past the fixed generation budget

        # --- draft phase: this_k sequential small forward passes, cache carried ---
        drafted = []
        for _ in range(this_k):
            next_token = draft_next_logits.argmax(dim=-1, keepdim=True)
            drafted.append(next_token)
            d_out = draft_model(input_ids=next_token, past_key_values=draft_cache, use_cache=True)
            draft_cache = d_out.past_key_values
            draft_next_logits = d_out.logits[:, -1, :]
        drafted = torch.cat(drafted, dim=1)  # [1, this_k]

        # --- verify phase: one batched target forward pass over the this_k tokens ---
        t_out = target_model(input_ids=drafted, past_key_values=target_cache, use_cache=True)
        target_cache_extended = t_out.past_key_values  # covers up through all this_k drafted tokens

        # predictions[j] is what the target predicts should come at drafted[j]'s position,
        # for j=0..this_k-1 (carried-over logits from before this round, then this round's
        # own output for j=1..this_k-1); predictions[this_k] is the target's prediction for
        # whatever comes *after* all this_k drafted tokens (used as the correction/bonus).
        predictions = torch.cat([target_next_logits.unsqueeze(1), t_out.logits], dim=1)  # [1, this_k+1, vocab]

        drafted_list = drafted[0].tolist()
        accepted = 0
        while accepted < this_k and predictions[0, accepted].argmax().item() == drafted_list[accepted]:
            accepted += 1

        extra_token = predictions[0, accepted].argmax().view(1, 1)  # correction or bonus, always present
        stats.accepted_per_round.append(accepted)
        stats.tokens_accepted_from_draft += accepted
        stats.tokens_proposed += this_k
        stats.rounds += 1

        # Roll back both caches to just the accepted prefix (crop() is a no-op if
        # accepted == this_k, since nothing needs discarding).
        keep_len = generated.shape[1] + accepted
        target_cache_extended.crop(keep_len)
        draft_cache.crop(keep_len)

        # accepted <= this_k <= remaining always, so this only ever trims the trailing
        # extra_token -- never the drafted/accepted portion -- and only in the specific
        # case where accepted == remaining (the last round's drafted tokens were all
        # accepted and would otherwise overshoot max_new_tokens by exactly one).
        new_tokens = torch.cat([drafted[:, :accepted], extra_token], dim=1)[:, :remaining]

        # If EOS appears anywhere in this round's emitted tokens, truncate right after
        # it -- matching what plain .generate() does -- rather than appending whatever
        # this round happened to also produce past that point.
        new_tokens_list = new_tokens[0].tolist()
        if eos_token_id is not None and eos_token_id in new_tokens_list:
            eos_pos = new_tokens_list.index(eos_token_id)
            new_tokens = new_tokens[:, : eos_pos + 1]
            hit_eos = True

        generated = torch.cat([generated, new_tokens], dim=1)
        n_new += new_tokens.shape[1]
        stats.tokens_generated = n_new

        if hit_eos or n_new >= max_new_tokens:
            # Budget exhausted (or the last emitted token was extra_token itself and got
            # trimmed off by the `[:, :remaining]` cap above) -- nothing more will be
            # generated, so there's no next round to fold `extra_token` into caches for.
            break

        # Fold `extra_token` into both caches (its own K/V was never computed -- it
        # only ever existed as a prediction, never as fed input) and get the target's
        # next carried-over prediction for the following round.
        t_step = target_model(input_ids=extra_token, past_key_values=target_cache_extended, use_cache=True)
        target_cache = t_step.past_key_values
        target_next_logits = t_step.logits[:, -1, :]
        d_step = draft_model(
            input_ids=_draft_safe(extra_token, draft_vocab_size), past_key_values=draft_cache, use_cache=True
        )
        draft_cache = d_step.past_key_values
        draft_next_logits = d_step.logits[:, -1, :]

    return generated, stats
