"""Train our controlled SuperBPE / BPE baselines at a matched vocab.

Runs gigatoken's ``train_superbpe`` and ``train_bpe`` on the same training
slice at identical ``--vocab``, then exports both as HF ``tokenizer.json``
files (ByteLevel BPE; the SuperBPE export lifts the whitespace pretokenizer
with ``use_regex=false`` so its superwords fire at encode time). Records
training wall-clock and superword stats into a manifest the efficiency,
throughput, and parity axes all read.

    uv run --no-project benchmarks/superbpe/train_baselines.py \
        --file ~/data/owt_train.txt --train-mb 500 --vocab 50000 --transition 40000

Defaults follow the plan (50k vocab, 40k transition => 10k superwords). Use
smaller values for a quick local run; note stage 2 is O(n) in unit length,
so keep --train-mb / --vocab modest without the full OWT corpus.
"""

from __future__ import annotations

import argparse
import os
import time

import common


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_corpus_args(p)
    p.add_argument("--vocab", type=int, default=50_000, help="target vocab size for both baselines (default: %(default)s)")
    p.add_argument("--transition", type=int, default=40_000, help="SuperBPE stage-1 -> stage-2 transition point (default: %(default)s)")
    p.add_argument("--max-unit-len", type=int, default=128, help="stage-2 unit cap in bytes (default: %(default)s)")
    p.add_argument("--tie-breaking", default="huggingface", choices=["huggingface", "lexicographic"])
    p.add_argument("--outdir", default=os.path.join(common.HERE, "artifacts"))
    p.add_argument("--skip-bpe", action="store_true", help="only train the SuperBPE baseline")
    # The stage-1 scheme has been selectable since the trainers gained a
    # `pretokenizer` argument; the default stays "gpt2" so the committed
    # artifacts and the published numbers keep reproducing. Axis 3 wants
    # "superbpe_stage1" instead -- the reference trainer's own stage-1 regex --
    # because a trainer comparison that differs in pretokenization is not
    # controlled, and `--suffix` keeps that run from overwriting the default one.
    p.add_argument("--pretokenizer", default="gpt2", help="stage-1 scheme (default: %(default)s)")
    p.add_argument("--suffix", default="", help="appended to the artifact/manifest filenames")
    args = p.parse_args()

    from gigatoken import train_bpe, train_superbpe

    sep = args.separator.encode("utf-8") if args.separator else None
    train, _eval, synthetic = common.load_corpus(args.file, args.train_mb, args.eval_mb, sep or common.DEFAULT_SEPARATOR)
    print(f"training on {len(train) / 1e6:.1f} MB ({'synthetic' if synthetic else args.file})")

    os.makedirs(args.outdir, exist_ok=True)
    def named(stem: str) -> str:
        return os.path.join(args.outdir, f"{stem}{args.suffix}.json")

    manifest: dict = {
        "corpus": {
            "file": args.file,
            "synthetic": synthetic,
            "train_mb": round(len(train) / 1e6, 2),
            "train_bytes": len(train),
            "separator": args.separator,
        },
        "settings": {
            "vocab": args.vocab,
            "transition": args.transition,
            "max_unit_len": args.max_unit_len,
            "tie_breaking": args.tie_breaking,
            "pretokenizer": args.pretokenizer,
        },
        "cpu": common.cpu_label(),
        "tokenizers": {},
    }

    # --- our SuperBPE -----------------------------------------------------
    # The two stages run inside one Rust call, so the split comes back through
    # the `timings` out-dict rather than a second perf_counter here; Axis 3
    # compares it against a reference that runs its stages as two processes.
    s_timings: dict[str, float] = {}
    t0 = time.perf_counter()
    s_vocab, s_merges = train_superbpe(
        train, args.vocab, args.transition, [],
        tie_breaking=args.tie_breaking, separator=sep, max_unit_len=args.max_unit_len,
        pretokenizer=args.pretokenizer, timings=s_timings,
    )
    s_time = time.perf_counter() - t0
    s_path = named("supergigatoken")
    common.save_hf_tokenizer(common.to_hf_bpe(s_vocab, s_merges, use_regex=False), s_path)
    stats = common.superword_stats(s_vocab)
    manifest["tokenizers"]["supergigatoken"] = {
        "path": s_path,
        "vocab_size": len(s_vocab),
        "train_time_s": round(s_time, 3),
        "stage1_time_s": round(s_timings["stage1_s"], 3) if "stage1_s" in s_timings else None,
        "stage2_time_s": round(s_timings["stage2_s"], 3) if "stage2_s" in s_timings else None,
        "use_regex": False,
        "pretokenizer": args.pretokenizer,
        **stats,
    }
    print(f"SuperBPE: {len(s_vocab)} tokens, {stats['n_superwords']} superwords, {s_time:.1f}s "
          f"(stage 1 {s_timings.get('stage1_s', float('nan')):.1f}s, stage 2 {s_timings.get('stage2_s', float('nan')):.1f}s) -> {s_path}")

    # --- our plain BPE (matched vocab) -----------------------------------
    if not args.skip_bpe:
        t0 = time.perf_counter()
        b_vocab, b_merges = train_bpe(
            train, args.vocab, [], tie_breaking=args.tie_breaking, separator=sep,
            pretokenizer=args.pretokenizer,
        )
        b_time = time.perf_counter() - t0
        b_path = named("gigatoken")
        common.save_hf_tokenizer(common.to_hf_bpe(b_vocab, b_merges, use_regex=True), b_path)
        manifest["tokenizers"]["gigatoken"] = {
            "path": b_path,
            "vocab_size": len(b_vocab),
            "train_time_s": round(b_time, 3),
            "use_regex": True,
            **common.superword_stats(b_vocab),
        }
        print(f"BPE:      {len(b_vocab)} tokens, {b_time:.1f}s -> {b_path}")

    manifest_path = named("baselines")
    common.save_json(manifest_path, manifest)
    print(f"wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
