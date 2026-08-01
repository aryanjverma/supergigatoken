"""Aggregate the three SuperBPE axes into one markdown report.

Reads the JSONs written by ``efficiency.py`` (Axis 1), ``throughput.py``
(Axis 2), and ``parity.py`` (Axis 3) and renders ``REPORT.md`` — an
efficiency table, a throughput table, and a trainer-parity table, plus
optional bar charts (matplotlib) in the visual style of the top-level
README's benchmark block. Each axis is optional; missing JSONs are noted so
the report still renders from whatever has been run.

    uv run --no-sync benchmarks/superbpe/efficiency.py --released
    uv run --no-sync benchmarks/superbpe/throughput.py --released
    uv run --no-sync benchmarks/superbpe/parity.py
    uv run --no-sync benchmarks/superbpe/report.py            # -> REPORT.md
"""

from __future__ import annotations

import argparse
import os

import common

GROUP_ORDER = {"ours": 0, "reference": 1, "repos": 2}
GROUP_LABEL = {"ours": "ours (matched vocab)", "reference": "released SuperBPE", "repos": "gigatoken benchmark set"}


def _meta_line(meta: dict) -> str:
    bits = []
    if meta.get("cpu"):
        bits.append(f"CPU: {meta['cpu']}")
    if meta.get("eval_mb") is not None:
        bits.append(f"eval slice: {meta['eval_mb']} MB")
    if meta.get("docs") is not None:
        bits.append(f"{meta['docs']} docs")
    if meta.get("synthetic"):
        bits.append("**synthetic corpus** (OWT file absent; numbers illustrative)")
    s = meta.get("settings") or {}
    if s:
        bits.append(f"vocab={s.get('vocab')}, transition={s.get('transition')}")
    return " · ".join(str(b) for b in bits)


def efficiency_section(data: dict, out: list[str]) -> None:
    out.append("## Axis 1 — Encoding efficiency (bytes/token)\n")
    if not data:
        out.append("_No `results_efficiency.json`; run `efficiency.py`._\n")
        return
    out.append(f"_{_meta_line(data.get('meta', {}))}_\n")
    out.append("Higher bytes/token = fewer tokens for the same text = more efficient. "
               "The controlled result is **our SuperBPE vs our BPE at identical vocab** "
               "(only the whitespace restriction differs); other rows are reference points "
               "at their own vocab sizes.\n")
    rows = sorted(
        data.get("tokenizers", {}).items(),
        key=lambda kv: (GROUP_ORDER.get(kv[1].get("group"), 9), -(kv[1].get("bytes_per_token") or 0)),
    )
    out.append("| Tokenizer | Group | Vocab | Bytes/token | Tokens |")
    out.append("|---|---|---:|---:|---:|")
    for name, rec in rows:
        out.append(f"| `{name}` | {GROUP_LABEL.get(rec.get('group'), rec.get('group'))} | "
                   f"{rec.get('vocab_size')} | **{rec.get('bytes_per_token')}** | {rec.get('tokens')} |")
    out.append("")
    ours = {n: r for n, r in data.get("tokenizers", {}).items() if r.get("group") == "ours"}
    sb = ours.get("ours_superbpe")
    bpe = ours.get("ours_bpe")
    if sb and bpe and bpe.get("bytes_per_token"):
        gain = sb["bytes_per_token"] / bpe["bytes_per_token"]
        tok_red = 1 - (sb.get("tokens", 0) / bpe["tokens"]) if bpe.get("tokens") else None
        line = f"At matched vocab, our SuperBPE reaches **{gain:.2f}x** the bytes/token of plain BPE"
        if tok_red is not None:
            line += f" — **{tok_red * 100:.1f}% fewer tokens** for the same text"
        out.append(line + ".\n")


def throughput_section(data: dict, out: list[str]) -> None:
    out.append("## Axis 2 — Encoding throughput (gigatoken vs HF)\n")
    if not data:
        out.append("_No `results_throughput.json`; run `throughput.py`._\n")
        return
    meta = data.get("meta", {})
    out.append(f"_{_meta_line(meta)}"
               + (f" · min of {meta.get('repeats')} repeats" if meta.get("repeats") else "")
               + "_\n")
    out.append("gigatoken fast-encodes a SuperBPE tokenizer via the `Superword` pretokenizer "
               "(whitespace lifted). tiktoken is skipped — it cannot represent SuperBPE.\n")
    out.append("| Tokenizer | gigatoken MB/s | HF MB/s | speedup | gigatoken Mtok/s | HF Mtok/s |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for name, rec in data.get("tokenizers", {}).items():
        g = rec.get("engines", {}).get("gigatoken", {})
        h = rec.get("engines", {}).get("hf", {})
        out.append(f"| `{name}` | **{g.get('mb_per_s', '-')}** | {h.get('mb_per_s', '-')} | "
                   f"{rec.get('speedup_vs_hf', '-')}x | {g.get('mtokens_per_s', '-')} | {h.get('mtokens_per_s', '-')} |")
    out.append("")


def parity_section(data: dict, out: list[str]) -> None:
    out.append("## Axis 3 — Trainer parity vs the original SuperBPE\n")
    if not data:
        out.append("_No `results_parity.json`; run `parity.py` (and `reference/run_reference.py` for the reference side)._\n")
        return
    out.append(f"_{_meta_line(data.get('meta', {}))}_\n")
    out.append("Outcome parity (training speed + tokenizer quality), **not** byte-identical "
               "merges — see `reference/README.md`.\n")
    out.append("| Trainer | Train s | Stage1 s | Stage2 s | Vocab | Superwords | Superword % | Bytes/token |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, rec in data.get("sides", {}).items():
        frac = rec.get("superword_fraction")
        out.append(f"| {rec.get('engine')} | {rec.get('train_time_s')} | {rec.get('stage1_time_s') or '-'} | "
                   f"{rec.get('stage2_time_s') or '-'} | {rec.get('vocab_size')} | {rec.get('n_superwords')} | "
                   f"{round(frac * 100, 2) if frac is not None else '-'} | {rec.get('bytes_per_token', '-')} |")
    out.append("")
    if "train_speedup_vs_reference" in data:
        out.append(f"gigatoken's `train_superbpe` trains in **{data['train_speedup_vs_reference']}x** "
                   "the reference's wall-clock (higher = faster).\n")
    ex = None
    for _, rec in data.get("sides", {}).items():
        if rec.get("superword_examples"):
            ex = rec["superword_examples"]
            break
    if ex:
        out.append("Example learned superwords: " + ", ".join(f"`{e}`" for e in ex[:8]) + ".\n")


def _plot_efficiency(data: dict) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    toks = data.get("tokenizers", {})
    if not toks:
        return None
    rows = sorted(toks.items(), key=lambda kv: kv[1].get("bytes_per_token") or 0)
    names = [n for n, _ in rows]
    vals = [r.get("bytes_per_token") or 0 for _, r in rows]
    colors = {"ours": "#2563eb", "reference": "#dc2626", "repos": "#9ca3af"}
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))))
    ax.barh(names, vals, color=[colors.get(toks[n].get("group"), "#999") for n in names])
    ax.set_xlabel("bytes / token (higher = more efficient)")
    ax.set_title("SuperBPE encoding efficiency")
    fig.tight_layout()
    path = os.path.join(common.HERE, "efficiency.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_throughput(data: dict) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    toks = data.get("tokenizers", {})
    names, giga, hf = [], [], []
    for name, rec in toks.items():
        g = rec.get("engines", {}).get("gigatoken", {})
        h = rec.get("engines", {}).get("hf", {})
        if g.get("mb_per_s") or h.get("mb_per_s"):
            names.append(name)
            giga.append(g.get("mb_per_s") or 0)
            hf.append(h.get("mb_per_s") or 0)
    if not names:
        return None
    import numpy as np

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(names))))
    ax.barh(y + 0.2, giga, height=0.4, label="gigatoken", color="#2563eb")
    ax.barh(y - 0.2, hf, height=0.4, label="HF tokenizers", color="#9ca3af")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("MB / s (higher = faster)")
    ax.set_title("SuperBPE encoding throughput")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(common.HERE, "throughput.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=common.REPORT_MD)
    p.add_argument("--no-plots", action="store_true", help="skip matplotlib bar charts")
    args = p.parse_args()

    eff = common.load_json(common.RESULTS_EFFICIENCY)
    thr = common.load_json(common.RESULTS_THROUGHPUT)
    par = common.load_json(common.RESULTS_PARITY)

    out: list[str] = []
    out.append("# SuperBPE evaluation (supergigatoken)\n")
    out.append("SuperBPE (Liu et al., 2025) trained natively by supergigatoken's `train_superbpe`, "
               "evaluated against the original released SuperBPE and gigatoken's benchmark "
               "tokenizer set along three axes: **encoding efficiency**, **encoding throughput**, "
               "and **trainer output/speed** vs the original reference.\n")

    plots: list[tuple[str, str | None]] = []
    if not args.no_plots:
        if eff:
            plots.append(("Encoding efficiency", _plot_efficiency(eff)))
        if thr:
            plots.append(("Encoding throughput", _plot_throughput(thr)))

    efficiency_section(eff, out)
    for title, path in plots:
        if title == "Encoding efficiency" and path:
            out.append(f"![{title}]({os.path.basename(path)})\n")

    throughput_section(thr, out)
    for title, path in plots:
        if title == "Encoding throughput" and path:
            out.append(f"![{title}]({os.path.basename(path)})\n")

    parity_section(par, out)

    out.append("---\n")
    out.append("_Regenerate: `efficiency.py`, `throughput.py`, `parity.py`, then `report.py`. "
               "The reference side of Axis 3 comes from `reference/run_reference.py` (isolated env)._\n")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
