"""Trainer throughput - gigatoken's BPE trainer vs HuggingFace ``tokenizers``.

The SuperBPE trainer already has a controlled comparison (Axis 3, against the
original SuperBPE implementation). This is the missing half: how the *plain*
BPE trainer, ``train_bpe``, compares to the trainer nearly everyone actually
uses. tiktoken is absent because it ships no trainer at all.

Controlled so the only difference is the implementation:

- same corpus bytes, same vocab size, same special tokens (none);
- both byte-level with the GPT-2 split regex -- ``train_bpe``'s default
  ``pretokenizer="gpt2"`` and HF's ``pre_tokenizers.ByteLevel``, which applies
  that same regex;
- HF's ``initial_alphabet`` seeded with all 256 byte characters, matching the
  byte-seeded vocabulary ``train_bpe`` starts from. Without this HF only learns
  the bytes its corpus happens to contain and would be solving a smaller
  problem;
- HF's ``min_frequency=0`` and ``max_token_length`` left unset, so neither side
  gets to prune the search the other cannot.

Both are handed the corpus as one in-memory buffer and timed with
``perf_counter`` around the train call only -- corpus loading, pretokenizer
construction and serialization are outside the timed region on both sides.

The default 100 MB matches Axis 3's slice, which is sized by what the *original
SuperBPE* implementation could survive (it reached 21 GB resident at 500 MB).
HF's BPE trainer has no such problem, so ``--train-mb`` can go higher; the
resident-memory column is recorded because a trainer that wins on time by
spending 10x the RAM has not obviously won.

    uv run --no-sync benchmarks/superbpe/trainer_vs_hf.py
    uv run --no-sync benchmarks/superbpe/trainer_vs_hf.py --train-mb 500 --repeats 1
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import common

RESULTS = os.path.join(common.HERE, "results_trainer.json")


def _peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MB, or None if unavailable.

    Windows exposes a true peak (``PeakWorkingSetSize``); elsewhere fall back to
    ``resource``, whose ``ru_maxrss`` is already a peak. Both are process-wide
    high-water marks, so a run must measure one trainer per process for the
    number to mean anything -- hence ``--engine`` below.
    """
    try:  # Windows
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return round(counters.PeakWorkingSetSize / 1e6, 1)
        return None
    except Exception:
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS bytes.
        return round((peak / 1e3 if peak > 1e6 else peak / 1e3), 1)
    except Exception:
        return None


def train_ours(corpus: bytes, vocab_size: int, sep: bytes) -> tuple[float, int]:
    from gigatoken import train_bpe

    t0 = time.perf_counter()
    vocab, _merges = train_bpe(
        corpus, vocab_size, [], tie_breaking="huggingface", separator=sep,
        pretokenizer="gpt2",
    )
    return time.perf_counter() - t0, len(vocab)


def train_hf(corpus: bytes, vocab_size: int, sep: bytes) -> tuple[float, int]:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    # Documents, not one blob: HF trains from an iterator of texts, and handing
    # it a single 100 MB string would make its pretokenizer walk one giant unit.
    # `train_bpe` splits on the same separator internally.
    docs = common.split_docs(corpus, sep)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=0,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    t0 = time.perf_counter()
    tok.train_from_iterator(docs, trainer=trainer, length=len(docs))
    return time.perf_counter() - t0, tok.get_vocab_size()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_corpus_args(p)
    p.add_argument("--vocab", type=int, default=50_000)
    p.add_argument("--repeats", type=int, default=3, help="min is kept (default: %(default)s)")
    p.add_argument(
        "--engine",
        choices=("both", "ours", "hf"),
        default="both",
        help="one engine per process when comparing peak RSS (default: %(default)s)",
    )
    p.add_argument("--out", default=RESULTS)
    args = p.parse_args()

    sep = common.DEFAULT_SEPARATOR
    train, _eval_slice, synthetic = common.load_corpus(
        args.file, args.train_mb, 0.0, separator=sep
    )
    n_bytes = len(train)
    print(
        f"train slice: {n_bytes / 1e6:.1f} MB, vocab={args.vocab}, "
        f"min of {args.repeats}{' (synthetic)' if synthetic else ''}"
    )

    engines = [("gigatoken", train_ours), ("hf", train_hf)]
    if args.engine != "both":
        engines = [e for e in engines if e[0] == ("gigatoken" if args.engine == "ours" else "hf")]

    out: dict = {
        "meta": {
            "corpus": args.file,
            "cpu": common.cpu_label(),
            "train_mb": round(n_bytes / 1e6, 2),
            "vocab": args.vocab,
            "repeats": args.repeats,
            "synthetic": synthetic,
            "note": "same corpus/vocab/pretokenizer; HF seeded with the full byte alphabet",
        },
        "engines": {},
    }
    for name, fn in engines:
        best = float("inf")
        measured = 0
        try:
            for _ in range(max(1, args.repeats)):
                gc.collect()
                elapsed, measured = fn(train, args.vocab, sep)
                best = min(best, elapsed)
        except Exception as ex:  # a trainer that OOMs or errors is recorded, not fatal
            out["engines"][name] = {"engine": name, "error": str(ex).splitlines()[0][:160]}
            print(f"  {name:<12} FAILED ({str(ex).splitlines()[0][:90]})")
            continue
        out["engines"][name] = {
            "engine": name,
            "train_s": round(best, 3),
            "mb_per_s": round(n_bytes / 1e6 / best, 2),
            "measured_vocab": measured,
            "peak_rss_mb": _peak_rss_mb(),
        }
        e = out["engines"][name]
        print(
            f"  {name:<12} {e['train_s']:>9.2f} s  {e['mb_per_s']:>7.2f} MB/s  "
            f"vocab={measured}  peak_rss={e['peak_rss_mb']} MB"
        )

    ours = out["engines"].get("gigatoken", {})
    hf = out["engines"].get("hf", {})
    if ours.get("train_s") and hf.get("train_s"):
        out["speedup_vs_hf"] = round(hf["train_s"] / ours["train_s"], 2)
        print(f"\n  train_bpe is {out['speedup_vs_hf']}x faster than HF tokenizers")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
