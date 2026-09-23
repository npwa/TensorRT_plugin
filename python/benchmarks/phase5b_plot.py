"""Phase 5B: render the per-prompt speedup chart (kernel vs. eager, same INT4 weights)
from build/phase5b/results.json -- individual chat-query speedups as bars, grouped and
colored by complexity category (short/medium/long, a sequential ramp since complexity is
ordered, not an arbitrary categorical split), with the mean speedup and its 95%
confidence interval overlaid so the reader sees both the average effect and how much it
actually varies prompt-to-prompt, not just a single summary number.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS_PATH = os.path.join(_PROJECT_ROOT, "build", "phase5b", "results.json")
OUT_PATH = os.path.join(_PROJECT_ROOT, "build", "phase5b", "speedup_chart.png")

PRIMARY = "#0B2038"
SECONDARY = "#1C7293"
ACCENT = "#F2A93B"
LIGHT_BG = "#F7F9FB"
TEXT_DARK = "#16232E"
TEXT_MUTED = "#5B6B7A"

CATEGORY_COLOR = {"short": "#8FC3D1", "medium": SECONDARY, "long": PRIMARY}
CATEGORY_ORDER = ["short", "medium", "long"]


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    per_prompt = sorted(
        data["per_prompt"], key=lambda r: (CATEGORY_ORDER.index(r["category"]), r["id"])
    )
    labels = [r["id"] for r in per_prompt]
    speedups = [r["speedup"] for r in per_prompt]
    colors = [CATEGORY_COLOR[r["category"]] for r in per_prompt]
    mean_speedup, ci_lo, ci_hi = data["mean_speedup"], data["ci95_lo"], data["ci95_hi"]

    fig, ax = plt.subplots(figsize=(12, 6.2), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    x = np.arange(len(labels))
    bars = ax.bar(x, speedups, color=colors, width=0.62, zorder=3)

    # 95% CI band for the mean, plus the mean line itself.
    ax.axhspan(ci_lo, ci_hi, color=ACCENT, alpha=0.18, zorder=1, label="95% CI of mean")
    ax.axhline(mean_speedup, color=ACCENT, linewidth=2.2, linestyle="--", zorder=2)
    ax.text(
        len(labels) - 0.4, mean_speedup + 0.15,
        f"mean = {mean_speedup:.2f}x  (95% CI {ci_lo:.2f}x–{ci_hi:.2f}x)",
        color="#B8790E", fontsize=11, fontweight="bold", ha="right", va="bottom",
    )

    for xi, v in zip(x, speedups):
        ax.text(xi, v + 0.12, f"{v:.1f}x", ha="center", va="bottom", fontsize=8.5, color=TEXT_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=TEXT_DARK)
    ax.set_ylabel("End-to-end speedup (kernel vs. eager INT4)", fontsize=11.5, color=TEXT_DARK)

    n_tokens = data.get("max_new_tokens")
    # Title and subtitle as separate figure-level texts (not ax.set_title + ax.text
    # sharing the same crowded region above the axes) so they get their own vertical
    # space and don't collide regardless of how tight_layout resolves the axes box.
    fig.suptitle(
        f"Phase 5B — Per-chat-query speedup across {len(labels)} prompts of varying complexity",
        fontsize=14.5, fontweight="bold", color=PRIMARY, x=0.03, y=0.985, ha="left",
    )
    fig.text(
        0.03, 0.945,
        f"Same quantized weights both configurations; {n_tokens} forced decode tokens per prompt; full generate() call, all 32 layers.",
        fontsize=9.5, color=TEXT_MUTED, ha="left", va="top",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D3DBE2")
    ax.spines["bottom"].set_color("#D3DBE2")
    ax.tick_params(colors=TEXT_MUTED)
    ax.grid(axis="y", color="#E1E7ED", linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOR[c]) for c in CATEGORY_ORDER]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=ACCENT, alpha=0.4))
    ax.legend(
        handles, [c.capitalize() + " prompt" for c in CATEGORY_ORDER] + ["95% CI of mean"],
        loc="upper left", frameon=False, fontsize=10, ncol=4, bbox_to_anchor=(0, -0.13),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT_PATH, dpi=150, facecolor=LIGHT_BG, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
