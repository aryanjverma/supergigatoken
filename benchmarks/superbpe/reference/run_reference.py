"""Run the *original* SuperBPE reference trainer (Liu et al., 2025).

This is the Axis 3 baseline: it drives PythonNut/superbpe's own two-stage
``train_tokenizer`` (stage 1 = whitespace-pretokenized BPE, stage 2 = resume
with the whitespace restriction lifted) on the *same* corpus slice and matched
settings as our ``gigatoken.train_superbpe``, then records the reference's
training wall-clock and emits a ``tokenizer.json``. ``report.py`` /
``parity.py`` on the gigatoken side read the manifest this writes.

IMPORTANT — isolated environment. The SuperBPE repo depends on a *fork* of
``huggingface/tokenizers`` (alisawuffles/tokenizers-superbpe) that conflicts
with the ``tokenizers`` gigatoken's dev env pins, so this script must be run
from the SuperBPE venv, NOT from ``uv run``. See README.md here for setup. It
imports nothing from gigatoken; it only shells out to the reference's
``python -m train_tokenizer`` and uses the stdlib.

Usage (inside the superbpe venv, with the repo cloned):

    python run_reference.py \
        --superbpe-repo /path/to/superbpe \
        --corpus ~/data/owt_train.txt --train-mb 500 \
        --vocab 50000 --transition 40000 \
        --outdir ./artifacts

The stage-1/stage-2 regexes default to the exact strings in the reference's
``scripts/train_tokenizer.sh`` and ``scripts/extend_tokenizer.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Verbatim from the reference scripts (train_tokenizer.sh / extend_tokenizer.sh).
STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
STAGE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"


def _utf8_boundary(data: bytes, idx: int) -> int:
    n = len(data)
    while idx < n and (data[idx] & 0xC0) == 0x80:
        idx += 1
    return min(idx, n)


def prepare_corpus(corpus: str, train_mb: float, work: Path) -> tuple[Path, int]:
    """Write the training slice into ``work/corpus/train/train.txt`` (the
    reference reads a directory of .txt files)."""
    corpus_dir = work / "corpus" / "train"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    want = int(train_mb * 1e6)
    src = os.path.expanduser(corpus)
    with open(src, "rb") as f:
        data = f.read(want + 8)
    data = data[: _utf8_boundary(data, min(want, len(data)))]
    out = corpus_dir / "train.txt"
    out.write_bytes(data)
    return corpus_dir, len(data)


def run_stage(python: str, repo: str, output_dir: Path, *, corpus_dir: Path | None, num_bytes: int | None, vocab_size: int, regex: str) -> float:
    """Invoke ``python -m train_tokenizer`` for one stage; return wall-clock.

    Every path handed to the subprocess must be absolute: it runs with
    ``cwd=repo`` (the reference imports ``train_tokenizer`` as a module, so it
    has to), and ``train_tokenizer.py`` then ``os.chdir``es into ``--output_dir``
    on top of that -- it looks for ``merges.txt`` in the working directory to
    decide whether it is extending. A relative path survives neither hop.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    cmd = [python, "-m", "train_tokenizer", "--output_dir", str(output_dir), "--vocab_size", str(vocab_size), "--regex_string", regex]
    if corpus_dir is not None:
        cmd += ["--corpus_dir", str(corpus_dir.resolve())]
    if num_bytes is not None:
        cmd += ["--num_bytes", str(num_bytes)]
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=repo, check=True)
    return time.perf_counter() - t0


def _load_vocab_bytes(tokenizer_json: Path) -> dict[int, bytes]:
    """Reconstruct id -> token bytes from a ByteLevel tokenizer.json vocab."""
    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]
    # invert GPT-2 byte<->unicode
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    out: dict[int, bytes] = {}
    for tok, tid in vocab.items():
        try:
            out[int(tid)] = bytes(u2b[ch] for ch in tok)
        except KeyError:
            out[int(tid)] = tok.encode("utf-8", "replace")
    return out


def _superword_stats(vocab: dict[int, bytes]) -> dict:
    def is_super(t: bytes) -> bool:
        return any(b == 0x20 and i > 0 and t[i - 1] != 0x20 for i, b in enumerate(t))

    supers = [t for t in vocab.values() if is_super(t)]
    examples = sorted(supers, key=len, reverse=True)[:12]
    return {
        "vocab_size": len(vocab),
        "n_superwords": len(supers),
        "superword_fraction": round(len(supers) / max(1, len(vocab)), 4),
        "superword_examples": [t.decode("utf-8", "replace") for t in examples],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--superbpe-repo", required=True, help="path to a clone of github.com/PythonNut/superbpe")
    p.add_argument("--python", default=sys.executable, help="interpreter of the superbpe venv (default: current)")
    p.add_argument("--corpus", default="~/data/owt_train.txt", help="training corpus (same slice as our train_baselines.py)")
    p.add_argument("--train-mb", type=float, default=500.0)
    p.add_argument("--vocab", type=int, default=50_000, help="final vocab size (matches our --vocab)")
    p.add_argument("--transition", type=int, default=40_000, help="stage-1 vocab / transition point (matches our --transition)")
    p.add_argument("--stage1-regex", default=STAGE1_REGEX)
    p.add_argument("--stage2-regex", default=STAGE2_REGEX)
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"))
    p.add_argument("--keep-work", action="store_true", help="keep the intermediate work dir")
    args = p.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.superbpe_repo))
    if not os.path.exists(os.path.join(repo, "train_tokenizer.py")):
        raise SystemExit(f"{repo} is not a superbpe checkout (no train_tokenizer.py); clone github.com/PythonNut/superbpe")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    corpus_dir, n_bytes = prepare_corpus(args.corpus, args.train_mb, work)
    print(f"reference corpus: {n_bytes / 1e6:.1f} MB -> {corpus_dir}")

    # Stage 1: whitespace-pretokenized BPE up to the transition vocab.
    stage1 = work / "stage1"
    t1 = run_stage(args.python, repo, stage1, corpus_dir=corpus_dir, num_bytes=n_bytes, vocab_size=args.transition, regex=args.stage1_regex)
    print(f"stage 1 ({args.transition} vocab): {t1:.1f}s")

    # Stage 2: inherit *all* stage-1 merges (our transition == stage-1 vocab),
    # reuse stage-1's corpus meta, and extend to the final vocab with the
    # whitespace restriction lifted (relaxed regex).
    stage2 = work / "stage2"
    stage2.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(stage1 / "merges.txt", stage2 / "merges.txt")
    shutil.copyfile(stage1 / "meta.json", stage2 / "meta.json")
    t2 = run_stage(args.python, repo, stage2, corpus_dir=None, num_bytes=None, vocab_size=args.vocab, regex=args.stage2_regex)
    print(f"stage 2 ({args.vocab} vocab): {t2:.1f}s")

    tok_json = stage2 / "tokenizer.json"
    final = outdir / "reference_superbpe.json"
    shutil.copyfile(tok_json, final)

    vocab = _load_vocab_bytes(tok_json)
    manifest = {
        "engine": "reference (PythonNut/superbpe)",
        "path": str(final),
        "corpus": args.corpus,
        "train_mb": round(n_bytes / 1e6, 2),
        "train_bytes": n_bytes,
        "settings": {
            "vocab": args.vocab,
            "transition": args.transition,
            "stage1_regex": args.stage1_regex,
            "stage2_regex": args.stage2_regex,
        },
        "stage1_time_s": round(t1, 3),
        "stage2_time_s": round(t2, 3),
        "train_time_s": round(t1 + t2, 3),
        **_superword_stats(vocab),
    }
    man_path = outdir / "reference_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nreference SuperBPE: {manifest['vocab_size']} tokens, {manifest['n_superwords']} superwords, "
          f"{manifest['train_time_s']:.1f}s total -> {final}")
    print(f"wrote manifest -> {man_path}")

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
