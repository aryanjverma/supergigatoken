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

```bash
# 1. clone the reference trainer
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git

# 2. dedicated venv (do NOT reuse the gigatoken uv env)
python3.12 -m venv .venv-superbpe
source .venv-superbpe/bin/activate      # Windows: .venv-superbpe\Scripts\activate

# 3. install the fork + reference deps
pip install -r requirements.txt
```

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
