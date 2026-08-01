# SuperBPE evaluation suite

Reproduces the [SuperBPE](../../README.md#superbpe) numbers in the top-level
README: it trains a SuperBPE tokenizer with supergigatoken's native
`train_superbpe`, then evaluates it against the original released SuperBPE and
gigatoken's benchmark tokenizer set along three axes.

| Axis | Script | Output |
|---|---|---|
| 1. Encoding efficiency (bytes/token) | `efficiency.py` | `results_efficiency.json` |
| 2. Encoding throughput (gigatoken vs HF) | `throughput.py` | `results_throughput.json` |
| 3. Trainer parity vs the original reference | `parity.py` (+ [`reference/`](reference/)) | `results_parity.json` |
| Aggregate report + plots | `report.py` | [`REPORT.md`](REPORT.md), `*.png` |
| Branded README figure | `plot_readme.py` | `../../assets/superbpe_efficiency.png` |

## Quick start

Get the corpus (the same OpenWebText sample gigatoken benchmarks on):

```bash
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz    # -> ~/data/owt_train.txt
```

Then run the pipeline (each script also runs on a built-in synthetic corpus if
the file is missing, so it works end-to-end offline):

```bash
uv run train_baselines.py --file ~/data/owt_train.txt   # our SuperBPE + matched plain BPE
uv run efficiency.py --released                          # bytes/token: ours + released + repo set
uv run throughput.py --released                          # MB/s: gigatoken vs HF
uv run parity.py                                          # trainer time + quality (our side)
uv run report.py                                          # aggregate REPORT.md + plots
uv run plot_readme.py                                     # branded assets/ figure
```

Common flags (see `--help`): `--train-mb` / `--eval-mb` (corpus split, default
500/100), `--vocab` / `--transition` (default 50000/40000), `--max-unit-len`.

## What each axis measures

- **Efficiency** — bytes/token on a disjoint held-out slice. The controlled
  result is *our SuperBPE vs our plain BPE at identical vocab* (only the
  whitespace restriction differs); the released 128k SuperBPE and the gigatoken
  repo set are reference points at their own vocab sizes.
- **Throughput** — gigatoken's `Superword` fast-encoder vs HuggingFace on the
  same slice (tiktoken can't represent SuperBPE). Both engines get the same
  pre-split document list.
- **Trainer parity** — training wall-clock and tokenizer quality vs the
  *original* SuperBPE trainer. That reference runs in its own isolated
  environment (a conflicting `tokenizers` fork); see [`reference/`](reference/).
  This is outcome parity, **not** byte-identical merges.

## Files

- `common.py` — shared corpus handling, gigatoken→HF conversion, metrics, JSON.
- `get_corpus.py` — optional streaming download of an OWT slice.
- `artifacts/` — trained `tokenizer.json`s + `baselines.json` manifest.
- `reference/` — isolated harness for the original SuperBPE trainer.
