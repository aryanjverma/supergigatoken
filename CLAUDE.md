# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of [gigatoken](https://github.com/marcelroed/gigatoken) — a Rust tokenizer (SIMD pretokenization, GB/s BPE encoding) exposed to Python via PyO3/maturin — extended with **SuperBPE**: `train_superbpe` (two-stage trainer), the `superbpe_stage1` and `superword` pretokenizer schemes, and `benchmarks/superbpe/`.

The Python package keeps the upstream name: the crate is `gigatoken` (lib `gigatoken_rs`), the extension module is `gigatoken.gigatoken_rs`, and users `import gigatoken`. Do not rename these — the fork is a strict superset, so all existing gigatoken code must keep working.

## Commands

Rust nightly is required (`#![feature(portable_simd)]`, plus `profile-rustflags` in `.cargo/config.toml`); `rust-toolchain.toml` pins it, so plain `cargo`/`uv` invocations get it automatically.

```bash
# Python: uv builds the Rust extension on first run and whenever any *.rs changes
uv run python -c "import gigatoken; print('ok')"

uv run pytest tests                        # full Python suite (~1470 tests)
uv run pytest tests/test_superbpe_train.py -q
uv run pytest tests/tokenizers/test_hf_parity.py -k gpt2
uv run ruff check . && uv run ruff format .
uv run ty check                             # type check (excludes notebooks/)

# Rust
cargo test                                  # unit tests live in #[cfg(test)] modules inside src/
cargo test --lib pretokenize::fast::superbpe_stage1
cargo bench --bench pretokenize             # criterion; also encode, encode_st, encode_doc, ...

# CLI (validate + time a HF repo against HF tokenizers)
uv run gigatoken bench openai-community/gpt2 owt_train.txt --validate --doc-separator "<|endoftext|>"
```

Test data: nothing large is committed. Python tests resolve HuggingFace files straight from the standard HF cache and download misses with `requests` (`tests/hf_cache.py`) — `huggingface_hub`/`transformers` are never imported just to fetch a file. Rust tests never download: they read the local HF cache and `#[ignore]`/skip when a repo is absent, except GPT-2, which falls back to `tests/fixtures/gpt2_tokenizer.json`. Large-corpus tests honor `OWT_MAX_BYTES` / `OWT_SLAB_BYTES`.

Windows is lightly tested upstream; prefer WSL for perf work.

## Architecture

Data flows **input → pretokenize → BPE merge → batch assembly → PyO3 bridge**.

- `src/pretokenize/` — the hot path and the bulk of the optimization work.
  - `fast/` holds the production scanners, one module per scheme, plus two shared const-generic families: `cl100k_family` and `o200k_family` (`<CONTRACTIONS, DIGITS3, SLASH, HAN>`). Most "new" schemes are an existing family instantiated with different const params — `superbpe_stage1` is `o200k_family::advance_pos::<false, true, true, false>`. `mask.rs` is the shared SIMD boundary scanner (NEON / runtime-detected AVX-512+AVX2) driving the two-phase chunk walker.
  - `options.rs` owns `PretokenizerType`: the scheme enum, `NAMES`, `from_name` (+ tiktoken aliases), and `from_split_regex` / `from_split_regexes`, which identify a scheme from the `Split` regexes in a HF `tokenizer.json`.
  - `reference/` (state machine, winnow combinator, portable-SIMD and AVX-512 prototypes) is **not** in the encode path. It exists as criterion baselines and as differential-test oracles. New schemes are validated against the reference regex, not against golden files.
- `src/bpe/` — the merge core, `pretoken_cache` (the cache hierarchy that makes warm encoding fast), `sentencepiece` (byte-fallback SPM BPE), `tiktoken`, and `superword` (SuperBPE two-level encoding, below).
- `src/bpe_train.rs` — `train_bpe` and `train_superbpe_stage2`. Stage 2 resumes from stage 1's vocab with pretoken boundaries removed, so a single token can span whitespace; it is O(n) in unit length (`max_unit_len` bounds it).
- `src/input/` — `file_source` (mmap, chunking at document boundaries), `jsonl`, `parquet`, `decompress` (.gz/.zst).
- `src/load_tokenizer/` — `hf` (tokenizer.json → BPE or SentencePiece), `tiktoken` (rank files, which carry no regex — the caller supplies the scheme name), `hub` (pure-filesystem HF cache resolution + direct download).
- `src/batch.rs` — parallel chunking, the pooled workers whose pretoken caches persist across calls, and the serial mirrors of every path. Every batch/file entry point has a `parallel=False` twin that never touches the process-global rayon pool.
- `src/bindings/` + `src/lib.rs` — PyO3 surface: `BPETokenizer`, `SentencePieceTokenizer`, sources, `train_bpe`/`train_superbpe`, padding/truncation, `PretokenizerIter`.
- `gigatoken/` (Python) — `_tokenizer.Tokenizer` is the single user-facing class; it picks the Rust backend automatically (SentencePiece when the model declares `byte_fallback`, byte-level BPE otherwise). `_hf_compat` / `_tiktoken_compat` are the `as_hf()` / `as_tiktoken()` drop-in adapters, `_load/` handles config-driven dispatch and Hub loading, `_cli.py` is the `gigatoken` Typer CLI.

Cross-cutting invariants worth knowing before editing:

- **Two module trees.** `src/main.rs` is a separate bin target with its own `mod` list (no `batch`, no `bindings`). Shared helpers must live where both trees see them — e.g. `madvise_hugepage` sits in `bpe/` for exactly this reason.
- **UTF-8 is trusted, not validated.** File contents and batch documents are assumed valid UTF-8 by documented contract; several `from_utf8_unchecked` calls depend on it. Argument-level checks (e.g. a non-UTF-8 separator on the SentencePiece path) stay.
- **Little-endian only** — key packing and token-lane stores `compile_error!` on big-endian.
- `parallel=False` paths must produce byte-identical output to the parallel ones, including when one huge document is split and reassembled.

### Adding a pretokenizer scheme

Touching one file is never enough. The `superbpe_stage1` commit (`00e61db`) is the reference example:

1. `src/pretokenize/fast/<scheme>.rs` — usually a `MaskScheme` impl delegating to a family with new const params.
2. `fast/mod.rs` — `pub mod` + `pub use`.
3. `options.rs` — enum variant, `pretokenize` dispatch arm, `NAMES` (bump the array length), `from_name`, and a `from_split_regex` arm so a `tokenizer.json` exported with that regex loads and fast-encodes.
4. `FastPretokenizerDispatch` variant.
5. Differential tests against the reference regex (small hand cases + randomized codepoint soup incl. combining marks), and register the scheme in the existing all-schemes cross-path tests.

### SuperBPE two-level encoding

`ByteLevel(use_regex=false)` means the whole document is one pretoken, which costs SuperBPE every mechanism gigatoken exists for. `src/bpe/superword.rs` recovers most of it: because merge priority is the token ID and stage-2 merges are appended after stage-1's, the merge table splits at a `threshold` into a stage-1 prefix that cannot span a stage-1 pretoken boundary and a superword suffix. Level 1 splits at stage-1 boundaries and encodes through the ordinary cached path with the prefix merges; level 2 runs the **full** table over the resulting token stream. Output is bit-identical.

Read that module's docs before touching it — the derivation has two non-obvious failure modes, both of which had to be found by measurement rather than reasoning:

- The threshold must be derived from a merge's **junction** (between its two operands), not from whether its whole bytes split. Byte-level BPE produces character *fragments* (`b"\xd0\xbe\xd0"`), and pretokenizing those describes input that cannot occur.
- The junction must be probed with the **actual token bytes**, not an abstracted character pair, and it must be probed against the *glued* splitter. Boundaries depend on how a run started.
- There is no sound "same character class ⇒ safe" shortcut: the o200k-family stage-1 regex splits `camelCase` between two letters and digit runs every three digits.

When verification fails, `enable_superword_two_level` installs no plan and the plain path runs — so a tokenizer the reasoning does not cover loses speed, never correctness. `superword_two_level_matches_single_pretoken` pins the equivalence; `bench_superword_two_level_vs_plain` (`--ignored`, needs OWT) measures both paths in one binary and asserts they agree over a 32 MB slice.

### SuperBPE specifics

- `train_bpe` / `train_superbpe` take a `pretokenizer=` scheme name, defaulting to `"gpt2"` so published OWT benchmark numbers stay reproducible. `FileSource`/parquet inputs *reject* a non-default scheme rather than silently ignoring it.
- The original SuperBPE stage-1 regex is the `superbpe_stage1` scheme, not `gpt2`. GPT-2's ` ?\p{L}+` excludes `\p{M}`, so combining marks fall out of the letter run — Devanagari `हिन्दी` becomes 6 pretokens and no consonant+matra unit can ever be learned (measured: −44.75% bytes/token for Hindi at 4k vocab). It also inflates the apparent superword gain, since stage 2 removes boundaries and repairs the damage in the SuperBPE arm only.
- Encoding a trained SuperBPE tokenizer requires exporting `tokenizer.json` with `ByteLevel(use_regex=False)`, which the loader maps to the `superword` scheme (a scheme that yields each segment whole). The released 128k SuperBPE ships an explicit `Split` regex that isn't mapped yet, so it can't be fast-encoded.

## Benchmarks

- `benchmarks/compare/` — cross-library throughput sweep. `sweep.py` dedups repos by tokenizer digest and runs one fresh process per measurement; `results.py merge` folds JSONL into `benchmarks/results.json` (best interleaved round, judged by gigatoken throughput) and `results.py render` rewrites the `<!-- benchmarks:start -->` block in `README.md`. Never hand-edit that block or `results.json`.
- `benchmarks/superbpe/` — the three-axis SuperBPE suite (efficiency / throughput / trainer parity) writing `results_*.json`, `REPORT.md`, and the README figures. Every script falls back to a synthetic corpus when the OWT file is missing, so it runs offline.
- `pretokenizer_optimization_log.md` and `profiling/` record the step-by-step perf history; consult them before re-litigating a micro-optimization.

## Conventions

- Rust comments are dense and explanatory: they justify *why* an unsafe block is sound (`// SAFETY:` on every one), why a hand-rolled sequence beats the obvious form, and what was measured. Match that density when editing hot code; include the measurement when claiming a speedup.
- `ruff` line-length is 180 with preview formatting; `[lints.rust] unused = "allow"`.
- `benches` inherit release codegen with debug info (profiling parity); the `profiling` profile adds frame pointers.
- Per `CONTRIBUTING.md`, upstream accepts issues rather than PRs; keep diffs concise and scoped.
