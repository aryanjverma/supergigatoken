# Axis 3 — original SuperBPE reference trainer

This directory runs the **original** SuperBPE trainer (Liu et al., 2025 —
[PythonNut/superbpe](https://github.com/PythonNut/superbpe)) so we can compare
our `gigatoken.train_superbpe` against it on **training wall-clock** and
**tokenizer quality** (bytes/token, superword rate), on the *same* corpus slice
and matched settings.

It lives in its own directory with its own environment because the reference
depends on a **fork** of `huggingface/tokenizers`
([alisawuffles/tokenizers-superbpe](https://github.com/alisawuffles/tokenizers-superbpe))
that conflicts with the `tokenizers` version gigatoken's dev env pins. Never
install these requirements into the gigatoken `uv` environment.

## Why a fork at all

Standard `tokenizers` cannot *resume* BPE training. SuperBPE's stage 2 resumes
from the stage-1 merges and continues learning with the whitespace restriction
lifted — that "resume + relaxed pretokenization regex" is exactly what the fork
adds. Our gigatoken implementation achieves the same outcome natively via
`train_superbpe` (stage-1 GPT-2 regex, then line-bounded stage-2 units).

## Setup (isolated venv)

Requires Python 3.12 and a Rust toolchain (the fork builds a native extension).
**3.12 exactly**: the fork is `tokenizers` 0.20.1 on pyo3 0.21, which has no
Python 3.13 support.

```bash
# 1. clone the reference trainer -- OUTSIDE this repository (see below)
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git ~/superbpe-reference/superbpe

# 2. dedicated venv (do NOT reuse the gigatoken uv env)
python3.12 -m venv .venv-superbpe
source .venv-superbpe/bin/activate      # Windows: .venv-superbpe\Scripts\activate

# 3. install the fork + reference deps
pip install -r requirements.txt
```

Verified on native Windows with `rustup`'s stable toolchain; no WSL needed.

### Clone it outside this repository

Cargo searches *upward* from the crate it is building for a workspace manifest.
A clone inside this repo finds gigatoken's own `Cargo.toml`, which uses the
nightly-only `profile-rustflags` feature, and the fork's build dies parsing it
(`the package requires the Cargo feature called 'profile-rustflags'`) before it
compiles a line. Keeping the checkout outside the tree avoids patching
third-party source; `--superbpe-repo` takes any path.

Correspondingly, pass **absolute** `--outdir` and `--python`: `run_reference.py`
invokes the reference with `cwd=<repo>` (it imports `train_tokenizer` as a
module) and `train_tokenizer.py` then `os.chdir`es into `--output_dir` on top of
that, because it looks for `merges.txt` in the working directory to decide
whether it is extending. The script resolves both for you now; a relative path
would survive neither hop.

### Memory: the reference does not scale to our committed 500 MB run

At 500 MB / 50k vocab / 40k transition on a 32 GB machine, reference stage 2
reached **21 GB resident after ~87 minutes and was still climbing**, heading for
the pagefile — abandoned, because a swapping process does not produce a
meaningful wall-clock. Stage 1 finished in ~60 s; stage 2 is the whole cost,
which is what lifting the whitespace restriction does to the trainer's unit set.

Measured stage-2 scaling at 5k vocab, one run each: 24.5 s at 20 MB, 56.2 s at
40 MB, 130.3 s at 80 MB — about 2.3x per doubling (≈ bytes^1.2), which
*understates* 500 MB, where memory growth dominates. Merge count barely matters:
at 20 MB, 8x the new merges (1.2k → 10k) cost 1.27x (24.5 → 31.1 s).

So Axis 3 is run at **100 MB**, where both sides complete. For a trainer
comparison the two sides matching *each other* is what makes it controlled; that
it is a smaller slice than the committed artifacts only means the Axis 3 numbers
are their own run, which the report's meta line states.

Beware run-to-run variance: a first measurement taken right after the Rust build
read 70.8 s where the same configuration reproducibly takes 24.5-26.3 s. Discard
the first run on a warm machine.

## Run

Match the settings you passed to `benchmarks/superbpe/train_baselines.py` so the
comparison is controlled (same corpus slice, same final vocab and transition):

```bash
python run_reference.py \
    --superbpe-repo ./superbpe \
    --corpus ~/data/owt_train.txt --train-mb 500 \
    --vocab 50000 --transition 40000 \
    --outdir ./artifacts
```

This drives the reference two-stage pipeline:

1. **Stage 1** — `python -m train_tokenizer` with the whitespace-splitting
   regex from the repo's `scripts/train_tokenizer.sh`, up to `--transition`
   vocab (equivalent to regular BPE).
2. **Stage 2** — inherit all stage-1 merges, then resume with the relaxed regex
   from `scripts/extend_tokenizer.sh`
   (`\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)`) up to `--vocab`,
   learning superwords.

## Outputs

- `artifacts/reference_superbpe.json` — the reference tokenizer.json.
- `artifacts/reference_manifest.json` — stage-1/stage-2/total wall-clock, vocab
  size, and superword stats (fraction + examples).

Feed both back to the gigatoken side:

```bash
# quality (bytes/token) alongside our SuperBPE and the repo set:
uv run --no-sync ../efficiency.py --released \
    --repos ""   # (or your set) and load the reference json via --manifest merge

# trainer parity (time + quality), and the combined report:
uv run --no-sync ../parity.py --reference artifacts/reference_manifest.json
uv run --no-sync ../report.py
```

## Not merge-identical parity

Our stage 1 is locked to the GPT-2 (r50k) regex and our stage 2 uses
line-bounded units, whereas the reference uses its own stage-1 regex and a
regex-driven stage-2 relaxation. So we compare **outcomes** (speed, bytes/token,
superword rate), **not** byte-identical merges. Merge-identical parity would
require parametrizing our stage-1 regex and adopting the reference stage-2 regex
(a possible follow-up).
