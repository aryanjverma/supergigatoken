"""Axis 2 - encoding throughput, the gigatoken way (gigatoken vs HF).

Mirrors ``benchmarks/compare/measure.py`` but for SuperBPE tokenizers, which
tiktoken cannot represent (its encoders bake in a whitespace-splitting regex,
so it is skipped here). For each SuperBPE tokenizer we measure both engines on
the same held-out eval slice and report MB/s and Mtok/s:

- gigatoken: loads the ``tokenizer.json`` and fast-encodes via the new
  ``PretokenizerType::Superword`` scheme (whitespace lifted; documents are not
  fragmented at interior whitespace, since superword merges bridge it);
- HF ``tokenizers``: loads the same ``tokenizer.json`` and encodes in parallel.

Both engines are handed the *same* pre-split document list (split on
``--separator``) so the comparison is apples-to-apples; unlike measure.py we do
not hand gigatoken one giant blob, because the Superword scheme deliberately
disables interior splitting and our saved tokenizers do not register the corpus
separator as a special token. Each engine is timed ``--repeats`` times and the
min is kept (least perturbed by GC / scheduler).

    uv run --no-sync benchmarks/superbpe/throughput.py --released
    uv run --no-sync benchmarks/superbpe/throughput.py --repeats 5

Tokenizers come from the ``train_baselines.py`` manifest (our SuperBPE; plain
BPE is included as a same-engine reference) plus, with ``--released``, a
released SuperBPE from the Hub. Every tokenizer is best-effort: a load/encode
failure on either engine is recorded and skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import common

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def _min_time(fn, repeats: int) -> tuple[float, int]:
    """Run ``fn`` ``repeats`` times; return (min elapsed seconds, token count)."""
    best = float("inf")
    tokens = 0
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        tokens = fn()
        best = min(best, time.perf_counter() - t0)
    return best, tokens


def measure_hf(path: str, docs: list[str], repeats: int) -> tuple[float, int]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    encode_batch = getattr(tok, "encode_batch_fast", tok.encode_batch)

    def run() -> int:
        return sum(len(e.ids) for e in encode_batch(docs))

    return _min_time(run, repeats)


def measure_gigatoken(path: str, docs_bytes: list[bytes], repeats: int) -> tuple[float, int]:
    import awkward as ak

    tok = gigatoken_tokenizer(path)

    def run() -> int:
        return int(ak.sum(ak.num(tok.encode_batch(docs_bytes, parallel=True))))

    return _min_time(run, repeats)


def gigatoken_tokenizer(path: str):
    import gigatoken

    return gigatoken.Tokenizer(path)


def _record(engine: str, elapsed: float, tokens: int, n_bytes: int) -> dict:
    return {
        "engine": engine,
        "tokens": tokens,
        "bytes": n_bytes,
        "time_s": round(elapsed, 4),
        "mb_per_s": round(n_bytes / 1e6 / elapsed, 2) if elapsed else None,
        "mtokens_per_s": round(tokens / 1e6 / elapsed, 3) if elapsed else None,
    }


def bench_tokenizer(name: str, path: str, docs: list[str], docs_bytes: list[bytes], n_bytes: int, repeats: int) -> dict:
    rec: dict = {"path": path, "engines": {}}
    for engine, fn, arg in (
        ("gigatoken", measure_gigatoken, docs_bytes),
        ("hf", measure_hf, docs),
    ):
        try:
            elapsed, tokens = fn(path, arg, repeats)
            rec["engines"][engine] = _record(engine, elapsed, tokens, n_bytes)
            e = rec["engines"][engine]
            print(f"  {name:<28} {engine:<9} {e['mb_per_s']:>8.2f} MB/s  {e['mtokens_per_s']:>7.3f} Mtok/s  ({tokens} tok)")
        except Exception as ex:
            rec["engines"][engine] = {"engine": engine, "error": str(ex).splitlines()[0][:120]}
            print(f"  {name:<28} {engine:<9} SKIPPED ({str(ex).splitlines()[0][:80]})", file=sys.stderr)
    g = rec["engines"].get("gigatoken", {})
    h = rec["engines"].get("hf", {})
    if g.get("mb_per_s") and h.get("mb_per_s"):
        rec["speedup_vs_hf"] = round(g["mb_per_s"] / h["mb_per_s"], 2)
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_corpus_args(p)
    p.add_argument("--manifest", default=os.path.join(common.HERE, "artifacts", "baselines.json"))
    p.add_argument("--released", action="store_true", help="also benchmark a released SuperBPE (cache-first via the Hub)")
    p.add_argument("--released-repo", default=common.RELEASED_SUPERBPE_REPO)
    p.add_argument("--repeats", type=int, default=3, help="timed repeats per engine; min is kept (default: %(default)s)")
    p.add_argument("--out", default=common.RESULTS_THROUGHPUT)
    args = p.parse_args()

    sep = args.separator.encode("utf-8") if args.separator else common.DEFAULT_SEPARATOR
    _train, ev, synthetic = common.load_corpus(args.file, args.train_mb, args.eval_mb, sep)
    docs = common.split_docs(ev, sep)
    docs_bytes = [d.encode("utf-8") for d in docs]
    n_bytes = sum(len(d) for d in docs_bytes)
    print(f"eval slice: {n_bytes / 1e6:.1f} MB, {len(docs)} docs, {args.repeats} repeats ({'synthetic' if synthetic else args.file})")

    manifest = common.load_json(args.manifest)
    results: dict = {
        "meta": {
            "corpus": args.file,
            "synthetic": synthetic,
            "eval_mb": round(n_bytes / 1e6, 2),
            "docs": len(docs),
            "repeats": args.repeats,
            "cpu": common.cpu_label(),
            "settings": manifest.get("settings", {}),
        },
        "tokenizers": {},
    }

    targets: list[tuple[str, str]] = []
    for name, info in manifest.get("tokenizers", {}).items():
        path = info.get("path")
        if path and os.path.exists(path):
            targets.append((name, path))
        else:
            print(f"skip {name}: artifact missing ({path}); run train_baselines.py first", file=sys.stderr)

    if args.released:
        try:
            from gigatoken.gigatoken_rs import hub_file

            targets.append((args.released_repo, str(hub_file(args.released_repo, "tokenizer.json"))))
        except Exception as ex:
            print(f"released SuperBPE: SKIPPED ({ex})", file=sys.stderr)

    for name, path in targets:
        results["tokenizers"][name] = bench_tokenizer(name, path, docs, docs_bytes, n_bytes, args.repeats)

    common.save_json(args.out, results)
    print(f"\nwrote {args.out}")
    print_table(results)


def print_table(results: dict) -> None:
    print("\n| Tokenizer | gigatoken MB/s | HF MB/s | speedup | gigatoken Mtok/s | HF Mtok/s |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, rec in results["tokenizers"].items():
        g = rec["engines"].get("gigatoken", {})
        h = rec["engines"].get("hf", {})
        print(
            f"| {name} | {g.get('mb_per_s', '-')} | {h.get('mb_per_s', '-')} | "
            f"{rec.get('speedup_vs_hf', '-')} | {g.get('mtokens_per_s', '-')} | {h.get('mtokens_per_s', '-')} |"
        )


if __name__ == "__main__":
    main()
