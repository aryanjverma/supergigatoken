"""Render the branded SuperBPE efficiency figure used in the README.

Reads ``results_efficiency.json`` (and, if present, ``results_throughput.json``)
and produces a two-panel PNG in ``assets/``:

- left: bytes/token vs vocabulary size (log-x) for every tokenizer measured —
  standard BPE/SentencePiece tokenizers trace a trend (more vocab -> more
  bytes/token), while SuperBPE tokenizers sit *above* that trend at their vocab
  size (superwords bridge whitespace);
- right: the controlled matched-vocab bar (~50k) — our SuperBPE vs our plain
  BPE vs same-size standard tokenizers.

    uv run --no-sync benchmarks/superbpe/plot_readme.py
"""

from __future__ import annotations

import argparse
import os

import common

BLUE = "#2563eb"
RED = "#dc2626"
GRAY = "#9aa3af"
DARK = "#111827"


def _annotate(ax, x, y, text, dx=8, dy=6, color=DARK):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=8.5, color=color, fontweight="bold")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(common.REPO_ROOT, "assets", "superbpe_efficiency.png"))
    p.add_argument("--throughput-out", default=os.path.join(common.REPO_ROOT, "assets", "superbpe_throughput.png"))
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eff = common.load_json(common.RESULTS_EFFICIENCY)
    toks = eff.get("tokenizers", {})
    if not toks:
        raise SystemExit("no results_efficiency.json; run efficiency.py first")
    meta = eff.get("meta", {})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.55, 1]})

    # ---- Panel A: bytes/token vs vocab size --------------------------------
    def pts(group):
        xs, ys, names = [], [], []
        for name, r in toks.items():
            if r.get("group") == group and r.get("vocab_size") and r.get("bytes_per_token"):
                xs.append(r["vocab_size"])
                ys.append(r["bytes_per_token"])
                names.append(name)
        return xs, ys, names

    rx, ry, _ = pts("repos")
    axL.scatter(rx, ry, s=46, c=GRAY, edgecolors="white", linewidths=0.6, label="standard tokenizers", zorder=3)

    for name, r in toks.items():
        v, b, g = r.get("vocab_size"), r.get("bytes_per_token"), r.get("group")
        if not (v and b):
            continue
        if name == "ours_superbpe":
            axL.scatter([v], [b], s=340, marker="*", c=BLUE, edgecolors="white", linewidths=1.1, zorder=6, label="supergigatoken (SuperBPE)")
            _annotate(axL, v, b, f"supergigatoken\n{b:.2f} B/tok @ {v // 1000}k", dx=10, dy=-2, color=BLUE)
        elif name == "ours_bpe":
            axL.scatter([v], [b], s=110, marker="o", c=BLUE, edgecolors="white", linewidths=1.0, zorder=6, label="gigatoken (plain BPE)")
            _annotate(axL, v, b, "gigatoken", dx=8, dy=-14, color=BLUE)
        elif g == "reference":
            axL.scatter([v], [b], s=340, marker="*", c=RED, edgecolors="white", linewidths=1.1, zorder=6, label="released SuperBPE (128k)")
            _annotate(axL, v, b, f"released SuperBPE\n{b:.2f} B/tok @ {v // 1000}k", dx=-140, dy=-6, color=RED)

    # label a couple of well-known standard points for context
    for tag in ("openai-community/gpt2", "google/gemma-4-E4B-it"):
        r = toks.get(tag)
        if r and r.get("vocab_size") and r.get("bytes_per_token"):
            short = tag.split("/")[-1]
            _annotate(axL, r["vocab_size"], r["bytes_per_token"], short, dx=6, dy=-13, color="#6b7280")

    axL.set_xscale("log")
    axL.set_xlabel("vocabulary size (log scale)")
    axL.set_ylabel("bytes / token  (higher = fewer tokens = more efficient)")
    axL.set_title("Encoding efficiency vs vocabulary size", fontweight="bold")
    axL.grid(True, which="both", axis="y", alpha=0.25)
    axL.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    # ---- Panel B: matched ~50k vocab bar -----------------------------------
    wanted = [
        ("ours_superbpe", "supergigatoken\n(50k)", BLUE),
        ("ours_bpe", "gigatoken\n(50k)", "#93c5fd"),
        ("openai-community/gpt2", "GPT-2\n(50k)", GRAY),
        ("answerdotai/ModernBERT-base", "ModernBERT\n(50k)", GRAY),
    ]
    labels, vals, colors = [], [], []
    for key, label, color in wanted:
        r = toks.get(key)
        if r and r.get("bytes_per_token"):
            labels.append(label)
            vals.append(r["bytes_per_token"])
            colors.append(color)
    bars = axR.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.0, zorder=3)
    for rect, v in zip(bars, vals):
        axR.text(rect.get_x() + rect.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axR.set_ylabel("bytes / token")
    axR.set_title("Same 50k vocab: superwords win", fontweight="bold")
    axR.grid(True, axis="y", alpha=0.25)
    axR.set_ylim(0, max(vals) * 1.18)

    sub = f"OpenWebText, {meta.get('eval_mb', '?')} MB held-out"
    if meta.get("synthetic"):
        sub += " (synthetic corpus — illustrative)"
    fig.suptitle(f"supergigatoken · SuperBPE encoding efficiency   —   {sub}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out}")

    _plot_throughput(plt, args.throughput_out)


_THR_LABELS = {"ours_superbpe": "supergigatoken\n(SuperBPE, 50k)"}


def _thr_label(name: str) -> str:
    if name in _THR_LABELS:
        return _THR_LABELS[name]
    return "released SuperBPE\n(128k)" if "superbpe" in name.lower() else name


def _plot_throughput(plt, out: str) -> None:
    """MB/s (gigatoken vs HF) for the SuperBPE tokenizers only."""
    import numpy as np

    thr = common.load_json(common.RESULTS_THROUGHPUT)
    toks = thr.get("tokenizers", {})
    # Only SuperBPE tokenizers our engine can actually encode (drops the plain
    # BPE row and any tokenizer gigatoken can't load, so no confusing n/a bars).
    names = [n for n in toks if "superbpe" in n.lower() and toks[n]["engines"].get("gigatoken", {}).get("mb_per_s")]
    names.sort(key=lambda n: (0 if n == "ours_superbpe" else 1, n))
    if not names:
        print("no encodable SuperBPE rows in results_throughput.json; skipping throughput figure")
        return

    labels = [_thr_label(n) for n in names]
    giga = [toks[n]["engines"].get("gigatoken", {}).get("mb_per_s") or 0 for n in names]
    hf = [toks[n]["engines"].get("hf", {}).get("mb_per_s") or 0 for n in names]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(5.5, 2.8 * len(names) + 2.4), 4.6))
    ax.set_xlim(-0.7, len(names) - 0.3)
    b1 = ax.bar(x - w / 2, giga, w, label="supergigatoken (gigatoken engine)", color=BLUE, edgecolor="white", zorder=3)
    b2 = ax.bar(x + w / 2, hf, w, label="HuggingFace tokenizers", color=GRAY, edgecolor="white", zorder=3)
    for rect, v in list(zip(b1, giga)) + list(zip(b2, hf)):
        txt = f"{v:.1f}" if v else "n/a"
        ax.text(rect.get_x() + rect.get_width() / 2, (v or 0) + max(giga + hf) * 0.01, txt, ha="center", va="bottom", fontsize=9, fontweight="bold")
    # speedup annotations where both engines ran
    for xi, g, h in zip(x, giga, hf):
        if g and h:
            ax.annotate(f"{g / h:.1f}× faster", (xi - w / 2, g * 0.5), textcoords="offset points", xytext=(-48, 0), ha="right", va="center", fontsize=10, fontweight="bold", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MB / s  (higher = faster)")
    ax.set_ylim(0, max(giga + hf) * 1.25)
    ax.set_title("supergigatoken · SuperBPE encoding throughput\n(same tokenizer, OWT 100 MB held-out, Intel 8-core)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
