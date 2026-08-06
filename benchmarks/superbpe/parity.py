"""Axis 3 - trainer parity: our train_superbpe vs the original SuperBPE.

Compares our ``gigatoken.train_superbpe`` against the reference trainer
(PythonNut/superbpe, run separately in its isolated env via
``reference/run_reference.py``) on:

- training wall-clock (same corpus slice, matched vocab + transition), and
- tokenizer quality: bytes/token on the held-out eval slice, superword rate,
  and superword examples.

Reads our ``train_baselines.py`` manifest and the reference manifest, measures
bytes/token for each side's tokenizer.json on the eval slice (best-effort — the
reference's forked pretokenizer may not load under vanilla ``tokenizers``), and
writes ``results_parity.json`` + a printed table.

This is outcome parity, NOT byte-identical merges: our stage 1 is locked to the
GPT-2 regex and our stage 2 uses line-bounded units, while the reference uses
its own regexes. See reference/README.md.

    uv run --no-sync benchmarks/superbpe/parity.py \
        --reference benchmarks/superbpe/reference/artifacts/reference_manifest.json
"""

from __future__ import annotations

import argparse
import os
import sys

import common


def _bytes_per_token(path: str, docs: list[str], n_bytes: int) -> tuple[float, int, int] | None:
    """(bytes/token, tokens, vocab_size) for a tokenizer.json, or None on failure."""
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(path)
        n_tokens = common.count_tokens_hf(tok, docs)
        try:
            vocab = tok.get_vocab_size()
        except Exception:
            vocab = len(tok.get_vocab())
        return common.bytes_per_token(n_bytes, n_tokens), n_tokens, vocab
    except Exception as e:
        print(f"  bytes/token unavailable for {path}: {str(e).splitlines()[0][:100]}", file=sys.stderr)
        return None


def _side(name: str, info: dict, docs: list[str], n_bytes: int) -> dict:
    rec = {
        "engine": name,
        "train_time_s": info.get("train_time_s"),
        "stage1_time_s": info.get("stage1_time_s"),
        "stage2_time_s": info.get("stage2_time_s"),
        "vocab_size": info.get("vocab_size"),
        "n_superwords": info.get("n_superwords"),
        "superword_fraction": info.get("superword_fraction"),
        "superword_examples": info.get("superword_examples"),
    }
    path = info.get("path")
    if path and os.path.exists(path):
        m = _bytes_per_token(path, docs, n_bytes)
        if m is not None:
            rec["bytes_per_token"], rec["tokens"], rec["measured_vocab"] = m
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_corpus_args(p)
    p.add_argument("--manifest", default=os.path.join(common.HERE, "artifacts", "baselines.json"), help="our train_baselines.py manifest")
    p.add_argument("--reference", default=os.path.join(common.HERE, "reference", "artifacts", "reference_manifest.json"), help="reference run_reference.py manifest")
    p.add_argument("--out", default=common.RESULTS_PARITY)
    args = p.parse_args()

    sep = args.separator.encode("utf-8") if args.separator else common.DEFAULT_SEPARATOR
    _train, ev, synthetic = common.load_corpus(args.file, args.train_mb, args.eval_mb, sep)
    docs = common.split_docs(ev, sep)
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"eval slice: {n_bytes / 1e6:.1f} MB, {len(docs)} docs ({'synthetic' if synthetic else args.file})")

    ours_manifest = common.load_json(args.manifest)
    ours_info = ours_manifest.get("tokenizers", {}).get("ours_superbpe", {})
    ref_info = common.load_json(args.reference)

    results: dict = {
        "meta": {
            "corpus": args.file,
            "synthetic": synthetic,
            "eval_mb": round(n_bytes / 1e6, 2),
            # The training slice both trainers were timed on, which is what the
            # wall-clock column means.
            "train_mb": (ours_manifest.get("corpus") or {}).get("train_mb"),
            "cpu": common.cpu_label(),
            "settings": ours_manifest.get("settings", {}),
            "note": "outcome parity (speed + quality), not byte-identical merges",
        },
        "sides": {},
    }

    if ours_info:
        results["sides"]["ours"] = _side("gigatoken train_superbpe", ours_info, docs, n_bytes)
        print(f"  ours:      {ours_info.get('train_time_s')}s, {ours_info.get('n_superwords')} superwords")
    else:
        print("no ours_superbpe in manifest; run train_baselines.py first", file=sys.stderr)

    if ref_info:
        results["sides"]["reference"] = _side("reference SuperBPE", ref_info, docs, n_bytes)
        print(f"  reference: {ref_info.get('train_time_s')}s, {ref_info.get('n_superwords')} superwords")
    else:
        print(
            f"no reference manifest at {args.reference}; run reference/run_reference.py in its "
            "isolated env first (see reference/README.md). Recording our side only.",
            file=sys.stderr,
        )

    o = results["sides"].get("ours", {})
    r = results["sides"].get("reference", {})
    if o.get("train_time_s") and r.get("train_time_s"):
        results["train_speedup_vs_reference"] = round(r["train_time_s"] / o["train_time_s"], 2)

    common.save_json(args.out, results)
    print(f"\nwrote {args.out}")
    print_table(results)


def print_table(results: dict) -> None:
    print("\n| Trainer | Train s | Stage1 s | Stage2 s | Vocab | Superwords | Superword % | Bytes/token |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, rec in results["sides"].items():
        frac = rec.get("superword_fraction")
        print(
            f"| {rec.get('engine')} | {rec.get('train_time_s')} | {rec.get('stage1_time_s') or '-'} | "
            f"{rec.get('stage2_time_s') or '-'} | {rec.get('vocab_size')} | {rec.get('n_superwords')} | "
            f"{round(frac * 100, 2) if frac is not None else '-'} | {rec.get('bytes_per_token', '-')} |"
        )
    if "train_speedup_vs_reference" in results:
        print(f"\ngigatoken train_superbpe is {results['train_speedup_vs_reference']}x the reference's training wall-clock (higher = faster).")


if __name__ == "__main__":
    main()
