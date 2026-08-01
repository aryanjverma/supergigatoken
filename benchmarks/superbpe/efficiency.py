"""Axis 1 - encoding efficiency (bytes/token) on a held-out slice.

The SuperBPE paper's headline metric, cross-tokenizer. Measures bytes per
token on the eval slice for:

- our trained SuperBPE and plain BPE at matched vocab (the controlled
  comparison: same corpus, same vocab, only the whitespace restriction
  differs) — loaded from the ``train_baselines.py`` artifacts;
- optionally the released 128k SuperBPE (``--released``) and the gigatoken
  benchmark tokenizer set (``--repos``) as external reference points at
  their own vocab sizes.

Higher bytes/token = fewer tokens for the same text = more efficient.

    uv run --no-sync benchmarks/superbpe/efficiency.py --released
    uv run --no-sync benchmarks/superbpe/efficiency.py --no-repos   # ours only

Our tokenizers are counted with HF ``tokenizers`` (the saved tokenizer.json
already bakes ByteLevel ``use_regex=false`` so superwords fire); the repo
set is counted with ``gigatoken.Tokenizer`` (handles BPE + SentencePiece and
uses gigatoken's own Hub loader). Token *counts* are engine-independent, so
mixing the two is fine. Every external tokenizer is best-effort: a load
failure (offline, rate-limited, unsupported) is recorded and skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import common


def _load_our(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def _vocab_size_hf(tok) -> int:
    try:
        return tok.get_vocab_size()
    except Exception:
        return len(tok.get_vocab())


def measure_hf(path: str, docs: list[str], n_bytes: int) -> dict:
    tok = _load_our(path)
    t0 = time.perf_counter()
    n_tokens = common.count_tokens_hf(tok, docs)
    return {
        "engine": "hf",
        "vocab_size": _vocab_size_hf(tok),
        "tokens": n_tokens,
        "bytes": n_bytes,
        "bytes_per_token": common.bytes_per_token(n_bytes, n_tokens),
        "encode_s": round(time.perf_counter() - t0, 3),
    }


def measure_gigatoken(repo: str, docs: list[str], n_bytes: int, local_file: str | None = None) -> dict:
    import gigatoken

    tok = gigatoken.Tokenizer(local_file) if local_file else gigatoken.Tokenizer(repo)
    t0 = time.perf_counter()
    n_tokens = common.count_tokens_gigatoken(tok, docs)
    vocab_size = getattr(tok, "vocab_size", None)
    if callable(vocab_size):
        vocab_size = vocab_size()
    return {
        "engine": "gigatoken",
        "vocab_size": vocab_size,
        "tokens": n_tokens,
        "bytes": n_bytes,
        "bytes_per_token": common.bytes_per_token(n_bytes, n_tokens),
        "encode_s": round(time.perf_counter() - t0, 3),
    }


def released_superbpe_path(repo: str) -> str:
    """Local tokenizer.json for a released SuperBPE (cache-first; downloads
    only on a cache miss)."""
    from gigatoken.gigatoken_rs import hub_file

    return str(hub_file(repo, "tokenizer.json"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_corpus_args(p)
    p.add_argument("--manifest", default=os.path.join(common.HERE, "artifacts", "baselines.json"))
    p.add_argument("--released", action="store_true", help="include a released SuperBPE reference (cache-first via the Hub)")
    p.add_argument("--released-repo", default=common.RELEASED_SUPERBPE_REPO, help="which released SuperBPE repo to load (default: %(default)s)")
    p.add_argument("--no-repos", action="store_true", help="skip the gigatoken benchmark tokenizer set")
    p.add_argument("--repos", nargs="*", default=None, help="override the benchmark repo list")
    p.add_argument("--plot", action="store_true", help="also write a bytes/token bar chart (needs matplotlib)")
    p.add_argument("--out", default=common.RESULTS_EFFICIENCY)
    args = p.parse_args()

    sep = args.separator.encode("utf-8") if args.separator else common.DEFAULT_SEPARATOR
    _train, ev, synthetic = common.load_corpus(args.file, args.train_mb, args.eval_mb, sep)
    docs = common.split_docs(ev, sep)
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"eval slice: {n_bytes / 1e6:.1f} MB, {len(docs)} docs ({'synthetic' if synthetic else args.file})")

    manifest = common.load_json(args.manifest)
    results: dict = {
        "meta": {
            "corpus": args.file,
            "synthetic": synthetic,
            "eval_mb": round(n_bytes / 1e6, 2),
            "docs": len(docs),
            "cpu": common.cpu_label(),
            "settings": manifest.get("settings", {}),
        },
        "tokenizers": {},
    }

    # --- our matched-vocab baselines -------------------------------------
    for name, info in manifest.get("tokenizers", {}).items():
        path = info.get("path")
        if not path or not os.path.exists(path):
            print(f"skip {name}: artifact missing ({path}); run train_baselines.py first", file=sys.stderr)
            continue
        try:
            rec = measure_hf(path, docs, n_bytes)
            rec["group"] = "ours"
            rec["train_time_s"] = info.get("train_time_s")
            rec["n_superwords"] = info.get("n_superwords")
            rec["superword_examples"] = info.get("superword_examples")
            results["tokenizers"][name] = rec
            print(f"  {name:<18} {rec['bytes_per_token']:.4f} B/tok  ({rec['tokens']} tok, vocab {rec['vocab_size']})")
        except Exception as e:
            print(f"  {name}: FAILED ({e})", file=sys.stderr)

    # --- released 128k SuperBPE ------------------------------------------
    if args.released:
        try:
            rec = measure_hf(released_superbpe_path(args.released_repo), docs, n_bytes)
            rec["group"] = "reference"
            results["tokenizers"][args.released_repo] = rec
            print(f"  {args.released_repo:<40} {rec['bytes_per_token']:.4f} B/tok  (vocab {rec['vocab_size']})")
        except Exception as e:
            print(f"  released SuperBPE: SKIPPED ({e})", file=sys.stderr)

    # --- gigatoken benchmark tokenizer set -------------------------------
    if not args.no_repos:
        repos = args.repos if args.repos is not None else common.benchmark_repos()
        for repo in repos:
            try:
                rec = measure_gigatoken(repo, docs, n_bytes)
                rec["group"] = "repos"
                results["tokenizers"][repo] = rec
                print(f"  {repo:<40} {rec['bytes_per_token']:.4f} B/tok  (vocab {rec['vocab_size']})")
            except Exception as e:
                print(f"  {repo}: SKIPPED ({str(e).splitlines()[0][:80]})", file=sys.stderr)

    common.save_json(args.out, results)
    print(f"\nwrote {args.out}")
    print_table(results)
    if args.plot:
        plot(results)


GROUP_ORDER = {"ours": 0, "reference": 1, "repos": 2}


def print_table(results: dict) -> None:
    rows = sorted(
        results["tokenizers"].items(),
        key=lambda kv: (GROUP_ORDER.get(kv[1].get("group"), 9), -(kv[1].get("bytes_per_token") or 0)),
    )
    print("\n| Tokenizer | Group | Vocab | Bytes/token | Tokens |")
    print("|---|---|---:|---:|---:|")
    for name, rec in rows:
        print(f"| {name} | {rec.get('group')} | {rec.get('vocab_size')} | {rec.get('bytes_per_token')} | {rec.get('tokens')} |")


def plot(results: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"plot skipped: matplotlib unavailable ({e})", file=sys.stderr)
        return
    rows = sorted(results["tokenizers"].items(), key=lambda kv: kv[1].get("bytes_per_token") or 0)
    names = [n for n, _ in rows]
    vals = [r.get("bytes_per_token") or 0 for _, r in rows]
    colors = {"ours": "#2563eb", "reference": "#dc2626", "repos": "#9ca3af"}
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))))
    ax.barh(names, vals, color=[colors.get(results["tokenizers"][n].get("group"), "#999") for n in names])
    ax.set_xlabel("bytes / token (higher = more efficient)")
    ax.set_title("SuperBPE encoding efficiency")
    fig.tight_layout()
    out = os.path.join(common.HERE, "efficiency.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
