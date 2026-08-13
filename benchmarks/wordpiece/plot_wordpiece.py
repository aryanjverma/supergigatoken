"""Render the WordPiece (BERT) figure used in the README.

Writes ``assets/wordpiece.png`` — two panels:

- **left**: encoding throughput against HuggingFace ``tokenizers`` on the same
  tokenizer and the same corpus slice;
- **right**: where a single-threaded encode's time actually goes, which is what
  says the scalar ``bert`` walker (not the normalizer, and not the merge core)
  is the remaining lever.

The two panels are *different measurements* and the subtitles say so: the left
is the parallel path across 8 cores, the right is one core, because a phase
split across rayon workers would attribute scheduling to whichever phase
happened to wait. Reading 618 and 219 as contradictory is the obvious trap, so
neither number appears without its condition.

Numbers are transcribed from the two commands recorded beside them rather than
re-measured here: this script draws, it does not benchmark. Re-run those and
update ``MEASURED`` to refresh the figure.

    uv run --no-sync benchmarks/wordpiece/plot_wordpiece.py
"""

from __future__ import annotations

import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Repo figure palette (benchmarks/superbpe/plot_readme.py) plus one hue the
# SuperBPE figures never needed. The three phase colors were checked with the
# dataviz validator rather than picked by eye: blue/amber/teal pass the
# lightness band, chroma floor, CVD separation (worst adjacent ΔE 12.5 protan,
# 28.7 tritan) and contrast. The obvious blue/violet/teal alternative FAILS at
# ΔE 0.4 for deuteranopia — the two would be one color to a red-green colorblind
# reader — and blue/teal/gray fails the chroma floor because gray reads as
# absence of category, not as a category.
BLUE = "#2563eb"
AMBER = "#d97706"
TEAL = "#0d9488"
GRAY = "#9aa3af"
DARK = "#111827"

MEASURED = {
    # benchmarks/compare/measure.py --library {gigatoken,hf}
    #   --tokenizer google-bert/bert-base-uncased
    #   --file ~/data/owt_train.txt --max-mb 100
    "throughput": {
        "corpus_mb": 100,
        "gigatoken_mb_s": 618.25,
        "hf_mb_s": 13.91,
        "cpu": "Intel 8-core",
    },
    # cargo test --release --lib bench_bert_phases -- --ignored --nocapture
    "phases": {
        "corpus_mb": 33.5,
        "docs": 6819,
        "full_mb_s": 219.3,
        # Shares of time-per-byte, which is what composes; a ratio of
        # throughputs does not.
        "parts": [
            ("pretokenize", 61.6, BLUE),
            ("normalize", 20.3, AMBER),
            ("cache + MaxMatch", 18.1, TEAL),
        ],
    },
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "assets", "wordpiece.png"))
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})

    _panel_throughput(axL)
    _panel_phases(axR)

    fig.suptitle(
        "supergigatoken · WordPiece (BERT) encoding",
        fontsize=13,
        fontweight="bold",
        color=DARK,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.text(
        0.5,
        0.012,
        "bert-base-uncased, OpenWebText · bit-identical to HuggingFace tokenizers 0.22.2 at add_special_tokens=False",
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out}")


def _panel_throughput(ax) -> None:
    """Two bars, 44× apart. The sliver *is* the finding, so the axis stays
    linear — a log scale would make a 44× gap look like a modest one."""
    t = MEASURED["throughput"]
    vals = [t["gigatoken_mb_s"], t["hf_mb_s"]]
    # Gray for HF is the repo's existing convention and a deliberate
    # de-emphasis. It is below the 3:1 contrast check, which obliges visible
    # labels — both bars carry their value, and identity here is positional
    # (axis categories), never color alone.
    colors = [BLUE, GRAY]
    bars = ax.bar([0, 1], vals, 0.55, color=colors, edgecolor="white", linewidth=2, zorder=3)
    for rect, v in zip(bars, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v + max(vals) * 0.02,
            f"{v:,.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=DARK,
        )
    # White, because this sits *inside* the blue bar. The first version drew it
    # in BLUE and it vanished — a palette validator checks the marks, not the
    # text drawn on top of them, which is why the procedure ends by looking at
    # the render.
    ax.text(
        0,
        t["gigatoken_mb_s"] * 0.5,
        f"{t['gigatoken_mb_s'] / t['hf_mb_s']:.0f}× faster",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="white",
        zorder=4,
    )
    ax.set_xlim(-0.62, 1.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["supergigatoken", "HuggingFace\ntokenizers"])
    ax.set_ylabel("MB / s  (higher = faster)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title(
        f"Throughput · {t['corpus_mb']} MB, {t['cpu']}, parallel",
        fontsize=10.5,
        color=DARK,
        pad=8,
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def _panel_phases(ax) -> None:
    """One 100% stacked bar: three parts of one whole, in pipeline order."""
    ph = MEASURED["phases"]
    left = 0.0
    for name, pct, color in ph["parts"]:
        ax.barh(
            0,
            pct,
            left=left,
            height=0.26,  # thin mark: the bar carries one number per segment
            color=color,
            edgecolor="white",
            linewidth=2,  # the 2px surface gap between adjacent fills
            zorder=3,
            label=name,
        )
        # Direct label inside each segment; every segment here is wide enough
        # (the narrowest is 18.1%) that no leader line is needed.
        ax.text(
            left + pct / 2,
            0,
            f"{pct:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
        left += pct

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("share of encode time")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title(
        f"Where the time goes · {ph['full_mb_s']} MB/s full encode\n({ph['corpus_mb']} MB, {ph['docs']:,} docs, single-threaded, min-of-5)",
        fontsize=10.5,
        color=DARK,
        pad=8,
    )
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    # Legend carries identity so it is never color-alone, and sits under the
    # bar rather than over the data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.30), ncol=3, fontsize=9.5, frameon=False)
    # Ink, not the series hue: text never carries identity here — its position
    # over the pretokenize segment does.
    ax.annotate(
        "the scalar walker is the remaining lever",
        (ph["parts"][0][1] / 2, 0.26),
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color=DARK,
    )


if __name__ == "__main__":
    main()
