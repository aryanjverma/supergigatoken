"""Render the branded SuperBPE figures used in the README.

Reads the ``results_*.json`` the other scripts in this directory write and
produces three PNGs in ``assets/``:

``superbpe_vs_original.png`` — the README's lead figure: us against the
*original* SuperBPE implementation (`results_parity.json` +
``results_vocab_diff.json``), three panels — training wall-clock, tokenizer
quality, and how much of the learned vocabulary the two agree on. Both sides
are the same corpus slice, vocab, transition point and stage-1 regex, which is
what makes it a comparison rather than two unrelated runs.

``superbpe_efficiency.png`` — two panels from ``results_efficiency.json``:

- left: bytes/token vs vocabulary size (log-x) for every tokenizer measured —
  standard BPE/SentencePiece tokenizers trace a trend (more vocab -> more
  bytes/token), while SuperBPE tokenizers sit *above* that trend at their vocab
  size (superwords bridge whitespace);
- right: the controlled matched-vocab bar (~50k) — supergigatoken's SuperBPE vs
  gigatoken's plain BPE vs same-size standard tokenizers.

``superbpe_throughput.png`` — encoding MB/s, our engine vs HuggingFace, from
``results_throughput.json``.

Any figure whose inputs are missing is skipped with a note rather than failing,
so this runs off whatever axes have been measured.

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
    p.add_argument("--vs-original-out", default=os.path.join(common.REPO_ROOT, "assets", "superbpe_vs_original.png"))
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_vs_original(plt, args.vs_original_out)

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
        if name == "supergigatoken":
            axL.scatter([v], [b], s=340, marker="*", c=BLUE, edgecolors="white", linewidths=1.1, zorder=6, label="supergigatoken (SuperBPE)")
            _annotate(axL, v, b, f"supergigatoken\n{b:.2f} B/tok @ {v // 1000}k", dx=10, dy=-2, color=BLUE)
        elif name == "gigatoken":
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
        ("supergigatoken", "supergigatoken\n(50k)", BLUE),
        ("gigatoken", "gigatoken\n(50k)", "#93c5fd"),
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


def _plot_vs_original(plt, out: str) -> None:
    """The lead figure: us vs the original SuperBPE implementation.

    Three panels, because a single one would be misleading in either direction.
    Training time alone reads as "8x faster" without saying what was given up;
    bytes/token alone hides that we got there in a fraction of the time; and
    both together still leave open whether the two trainers even learned the
    same thing, which is what the agreement panel answers.
    """
    par = common.load_json(common.RESULTS_PARITY)
    vd = common.load_json(os.path.join(common.HERE, "results_vocab_diff.json"))
    ours = (par.get("sides") or {}).get("ours") or {}
    ref = (par.get("sides") or {}).get("reference") or {}
    if not (ours.get("train_time_s") and ref.get("train_time_s")):
        print("no reference side in results_parity.json; skipping the vs-original figure")
        return

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.5, 4.9), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    labels = ["supergigatoken", "original\nSuperBPE"]
    colors = [BLUE, RED]

    # ---- Panel A: training wall-clock (lower = better) ---------------------
    times = [ours["train_time_s"], ref["train_time_s"]]
    bars = axA.bar(labels, times, color=colors, edgecolor="white", linewidth=1.0, zorder=3, width=0.62)
    for rect, v in zip(bars, times):
        axA.text(rect.get_x() + rect.get_width() / 2, v + max(times) * 0.02, f"{v:.0f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axA.set_ylim(0, max(times) * 1.24)
    axA.set_ylabel("training wall-clock, seconds  (lower = faster)")
    axA.set_title("Training time", fontweight="bold")
    axA.grid(True, axis="y", alpha=0.25)
    axA.annotate(
        f"{ref['train_time_s'] / ours['train_time_s']:.1f}× faster",
        (0, times[0]),
        textcoords="offset points",
        xytext=(0, 30),
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=BLUE,
    )

    # ---- Panel B: tokenizer quality (higher = better) ----------------------
    bpt = [ours.get("bytes_per_token"), ref.get("bytes_per_token")]
    if all(bpt):
        bars = axB.bar(labels, bpt, color=colors, edgecolor="white", linewidth=1.0, zorder=3, width=0.62)
        for rect, v in zip(bars, bpt):
            axB.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        # Zoomed y-axis would exaggerate a 3% gap; start at zero and say so.
        axB.set_ylim(0, max(bpt) * 1.22)
        axB.set_ylabel("bytes / token  (higher = more efficient)")
        axB.set_title("Tokenizer quality", fontweight="bold")
        axB.grid(True, axis="y", alpha=0.25)
        axB.annotate("no quality traded away", (0.5, max(bpt) * 1.1), xycoords=("axes fraction", "data"), ha="center", fontsize=10, fontweight="bold", color=DARK)

    # ---- Panel C: how much of the vocabulary the two agree on --------------
    d = vd.get("differential") or {}
    rows = [
        ("whole vocabulary", (d.get("vocab") or {}).get("jaccard")),
        ("subwords", (d.get("subwords") or {}).get("jaccard")),
        ("superwords", (d.get("superwords") or {}).get("jaccard")),
        ("merge order (Spearman)", (d.get("merges") or {}).get("rank_spearman")),
    ]
    rows = [(lab, v) for lab, v in rows if v is not None]
    if rows:
        names = [lab for lab, _ in rows][::-1]
        vals = [v for _, v in rows][::-1]
        # The Spearman row measures agreement on *priority*, not membership, so
        # it gets its own colour rather than reading as a fourth overlap.
        cols = ["#7c3aed" if "Spearman" in n else BLUE for n in names]
        bars = axC.barh(names, vals, color=cols, edgecolor="white", linewidth=1.0, zorder=3, height=0.6)
        for rect, v in zip(bars, vals):
            axC.text(v + 0.012, rect.get_y() + rect.get_height() / 2, f"{v:.3f}", va="center", fontsize=10.5, fontweight="bold")
        axC.set_xlim(0, 1.13)
        axC.set_xlabel("agreement with the original (1.0 = identical)")
        axC.set_title("They learn nearly the same tokenizer", fontweight="bold")
        axC.grid(True, axis="x", alpha=0.25)
    else:
        axC.axis("off")

    meta = par.get("meta") or {}
    s = meta.get("settings") or {}
    mb = meta.get("train_mb")
    mb = f"{mb:g}" if isinstance(mb, (int, float)) else "?"
    vocab = s.get("vocab")
    vocab = f"{vocab // 1000}k" if isinstance(vocab, int) and vocab >= 1000 else str(vocab)
    trans = s.get("transition")
    trans = f"{trans // 1000}k" if isinstance(trans, int) and trans >= 1000 else str(trans)
    # Title and provenance on separate lines: as one string this overflowed the
    # figure width and clipped at both ends.
    fig.suptitle("supergigatoken vs the original SuperBPE", fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.925,
        f"same {mb} MB OpenWebText slice · vocab {vocab} · transition {trans} · "
        f"identical stage-1 regex ({s.get('pretokenizer', '?')})",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


_THR_LABELS = {"supergigatoken": "supergigatoken\n(SuperBPE, 50k)"}


def _is_superbpe(name: str) -> bool:
    """Is this row a SuperBPE tokenizer (ours or a released one)?

    Substring matching alone no longer decides it: our row is named
    `supergigatoken`, which does not contain "superbpe", while `gigatoken`
    (plain BPE) is a substring of it.
    """
    return name == "supergigatoken" or "superbpe" in name.lower()


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
    names = [n for n in toks if _is_superbpe(n) and toks[n]["engines"].get("gigatoken", {}).get("mb_per_s")]
    names.sort(key=lambda n: (0 if n == "supergigatoken" else 1, n))
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
