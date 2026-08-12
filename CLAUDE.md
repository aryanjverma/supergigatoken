# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of [gigatoken](https://github.com/marcelroed/gigatoken) — a Rust tokenizer (SIMD pretokenization, GB/s BPE encoding) exposed to Python via PyO3/maturin — extended with **SuperBPE**: `train_superbpe` (two-stage trainer), the `superbpe_stage1`, `superword` and `superword_bounded` pretokenizer schemes, and `benchmarks/superbpe/`.

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
  - `fast/` holds the production scanners, one module per scheme, plus two shared const-generic families: `cl100k_family` and `o200k_family` (`<CONTRACTIONS, DIGITS3, SLASH, HAN>`). Most "new" schemes are an existing family instantiated with different const params — `superbpe_stage1` is `o200k_family::advance_pos::<false, true, true, false>`. `mask.rs` is the shared SIMD boundary scanner (NEON / runtime-detected AVX-512+AVX2) driving the two-phase chunk walker. `superword_bounded.rs` is the exception: a plain scalar walker, because the released 128k's regex is not any family's instantiation and its boundaries are not a subset of stage 1's, so the mask harvest cannot be filtered down to it (`"a \t b"` has an official boundary at offset 2 that stage 1's `{1,3}` never produces). Measured at 578.6 MB/s and 15% of the released 128k's two-level cost, so vectorizing it caps at 1.18×. `bert.rs` is the second scalar walker (no regex at all — two hardcoded splits), and the only scheme whose spans **do not partition the input**: whitespace is dropped, which `fill_spans_keyed_with_buf` permits (its contract is only `start < end` and in bounds) but every other scheme happens to satisfy.
  - `options.rs` owns `PretokenizerType`: the scheme enum, `NAMES`, `from_name` (+ tiktoken aliases), and `from_split_regex` / `from_split_regexes`, which identify a scheme from the `Split` regexes in a HF `tokenizer.json`.
  - `level1.rs` is the SuperBPE level-1 unit splitter: the glue predicate, a one-unit-at-a-time walker, and `Level1Fill`, which runs `MaskState::fill_spans_two_phase` with `GLUE = true` so level 1 gets the same SIMD harvest + branch-free emission the subword path uses. `GLUE` inserts one in-place filter over the harvested boundary buffer and const-folds away everywhere else.
  - `reference/` (state machine, winnow combinator, portable-SIMD and AVX-512 prototypes) is **not** in the encode path. It exists as criterion baselines and as differential-test oracles. New schemes are validated against the reference regex, not against golden files.
- `src/bpe/` — the merge core, `pretoken_cache` (the cache hierarchy that makes warm encoding fast), `sentencepiece` (byte-fallback SPM BPE), `tiktoken`, `superword` (SuperBPE two-level encoding, below), and `wordpiece` + `bert_normalizer` (BERT family, below).
- `src/bpe_train.rs` — `train_bpe` and `train_superbpe_stage2`. Stage 2 resumes from stage 1's vocab with pretoken boundaries removed, so a single token can span whitespace; it is O(n) in unit length (`max_unit_len` bounds it).
- `src/input/` — `file_source` (mmap, chunking at document boundaries), `jsonl`, `parquet`, `decompress` (.gz/.zst).
- `src/load_tokenizer/` — `hf` (tokenizer.json → BPE, WordPiece, or SentencePiece; `probe_model_family` decides, and only Unigram is still refused by name), `tiktoken` (rank files, which carry no regex — the caller supplies the scheme name), `hub` (pure-filesystem HF cache resolution + direct download).
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

A scheme whose pre_tokenizer carries no regex (`bert`) skips step 3's `from_split_regex` arm and is detected by kind name in `load_tokenizer::hf::detect_pretokenizer_type` instead.

### WordPiece / BERT

`bpe::wordpiece` is a third algorithm arm inside `bpe::tiktoken::Tokenizer`, not a backend: `encode_pretoken_miss` dispatches to MaxMatch on an `Option` field exactly like the `ranked_merges` arm, so the pretoken cache, `batch.rs`'s worker pool and serial mirrors, the PyO3 surface, and padding/truncation are all inherited unchanged — and the Python side needed no dispatch change at all. `bpe::bert_normalizer` materializes HF's `BertNormalizer` per segment in `for_each_piece`, next to the NFC hook. Design and the measured HF semantics: `docs/superpowers/specs/2026-08-10-fast-wordpiece-design.md`.

Three things here were established by measurement and will look wrong to anyone reasoning from Unicode alone:

- **HF's Unicode predicates are not ICU's, in three places.** `tokenizers` bundles an older UCD, so: 141 codepoints are `\p{P}` to ICU but word characters to HF (U+09FD Bengali, U+0C77 Telugu, U+2E43–2E5D, …) with 2 going the other way (U+166D `So`, U+111C9 `Mn`); `clean_text`'s control set differs by 20 `Cf` codepoints; and **`strip_accents`' mark set differs by 494** — ICU calls 2059 codepoints `Mn`, HF strips 1567. The mark delta is the dangerous one, because stripping is a *deletion*: classifying a codepoint as a mark that HF keeps drops it from the text entirely. U+111C9 must be **punctuation** to the splitter and **kept** by the normalizer simultaneously. All three delta lists are hardcoded in `pretokenize::unicode` and the class **populations are pinned in tests** (726 punct / 25 whitespace / 137681 removed / 1567 marks), so bumping `icu` fails loudly instead of mis-encoding multilingual text. Compute a delta against ICU's *own* set — deriving it from a third UCD silently misses everything that UCD leaves unassigned (that mistake cost 14 punctuation codepoints on the first attempt). The end-to-end guard is `test_bert_matches_hf_for_every_codepoint`, which encodes `"a<c>a"` for all 1.1M scalars through both libraries.
- **`Tokenizer::new_wordpiece` takes the model at construction**, because `from_tables` seeds the pretoken cache *during* construction and seeded entries are hits the miss path never revisits. Seeding routes through the same `encode_unit` as a cold miss, so the two cannot disagree.
- **The byte-wise MaxMatch backoff is exact, not approximate.** Vocab keys are valid UTF-8, so no key ends mid-character and a non-char-boundary probe simply fails — which is what HF's own loop does. No char-boundary walking anywhere.

Decoding is **not** invertible (the pretokenizer drops whitespace; the normalizer folds case and accents), so `TokenizerSpec.lossless_decode` marks those specs out of the shared roundtrip test and `tests/tokenizers/test_wordpiece.py` compares against HF's `WordPiece` decoder instead.

The normalizer is the cost centre, not the model: materialising it naively cost **62.6% of end-to-end encode** (`clean_text` alone 47.2%, NFD only 0.2%), fixed to 27.4% — encode 101.4 → 190.5 MB/s — by bulk-copying runs of printable ASCII in both the clean and fold passes instead of walking characters. A *whole-segment* ASCII precheck is useless (one newline disqualifies a 1 MB document); the win only exists per run. Both bulk paths are licensed by `bert_normalizer_matches_reference_random`, which fuzzes them against a four-pass reference kept in the file for that purpose. Numbers, the two failed attempts, and the still-deferred raw-keyed redesign are in `pretokenizer_optimization_log.md`.

### SuperBPE two-level encoding

`ByteLevel(use_regex=false)` means the whole document is one pretoken, which costs SuperBPE every mechanism gigatoken exists for. `src/bpe/superword.rs` recovers most of it: because merge priority is the token ID and stage-2 merges are appended after stage-1's, the merge table splits at a `threshold` into a stage-1 prefix that cannot span a stage-1 pretoken boundary and a superword suffix. Level 1 splits at stage-1 boundaries and encodes through the ordinary cached path with the prefix merges; level 2 runs the **full** table over the resulting token stream. Output is bit-identical.

Read that module's docs before touching it — the derivation has two non-obvious failure modes, both of which had to be found by measurement rather than reasoning:

- The threshold must be derived from a merge's **junction** (between its two operands), not from whether its whole bytes split. Byte-level BPE produces character *fragments* (`b"\xd0\xbe\xd0"`), and pretokenizing those describes input that cannot occur.
- The junction must be probed with the **actual token bytes**, not an abstracted character pair, and it must be probed against the *glued* splitter. Boundaries depend on how a run started.
- There is no sound "same character class ⇒ safe" shortcut: the o200k-family stage-1 regex splits `camelCase` between two letters and digit runs every three digits.

`glues` is two rule sets, not one. Whitespace runs, apostrophes, digit-initial right sides and `non_ascii_pair` are unconditional; word-initial right sides and whitespace-after-`\p{M}` are the `wide` set, which the *plan derives* rather than always applying — it is what the released 128k needs to reach 85956 (`wide` alone took it 485 → 13471, and `non_ascii_pair` the rest of the way) but costs 9.8% throughput (147.9 vs 163.9 MB/s) on a tokenizer that does not need it, so `build_capped` tries both per candidate scheme and keeps `wide` only when it strictly wins. Gluing more is always sound (it only removes level-1 split points, and removing all of them is the plain path) and never free.

`non_ascii_pair` is the exception that is unconditional for *correctness*, not speed. A merge's operands need not be whole characters — byte-level BPE merges character fragments — so a junction that yields no decodable right-hand character is not thereby interior: the released 128k's merge 13471 is `"ا"` + `b"\xd8"`, a real boundary whose following character is merely unknown. Reading that as safe made both the release and the committed 50k artifact mis-encode Arabic letter + ARABIC COMMA (5 tokens in 16,007,082 on the OWT eval slice). `Junction` in `superword.rs` separates "inside a character" from "boundary, character unknown", `open_right_can_split` enumerates the completions of a truncated head, and `non_ascii_pair` glues the rest with two byte comparisons — which is what keeps the threshold at 85956 instead of 13471. The remaining gap to the release's semantic transition (100164) is one genuinely unsafe merge, `b"\x8a"` + `b"\n"`; `open_left_cap_is_a_real_boundary` pins that so the conservative `Junction::OpenLeft` arm is known to be exact and not just safe.

Level 1 pulls its units through `pretokenize/fast/level1.rs` (above); `GIGATOK_SUPERWORD_L1FILL=iter|buf|twophase` selects the fill shape and `bench_superword_variants` A/Bs all three in one process. `twophase` is the default at +14.3% over the iterator alongside the cuts (156.3 vs 136.7 MB/s, interleaved min-of-5, 33.5 MB OWT). Two traps worth knowing: a pure-`advance` walk is *not* the mask scanner's partition on invalid UTF-8 (measured 750/2000 random byte soups, 0/2000 valid ones), so level-1 walkers must go through `next_span`; and the two-phase fill's `scan` cursor has to be rewound onto `pending`'s grid batch at every refill iteration, not just at fill entry, because gluing discards harvested boundaries on nearly every pass.

When verification fails, `enable_superword_two_level` installs no plan and the plain path runs — so a tokenizer the reasoning does not cover loses speed, never correctness. `superword_two_level_matches_single_pretoken` pins the equivalence; `bench_superword_two_level_vs_plain` (`--ignored`, needs OWT) measures both paths in one binary and asserts they agree over a 32 MB slice. `released_128k_two_level_matches_plain` and `bench_released_128k_vs_plain` are the twins for the released checkpoint (85.8 vs 31.6 MB/s, 2.72×, at threshold 85956). Each of those benches keeps two 128k-class tokenizers resident, so **exactly one may run per process** — that is why they are separate `#[test]`s with non-overlapping name prefixes rather than one parameterized test, and why `--ignored` runs must name a single bench. `bench_released_128k_phases` skips the level-2 replay arm: under `superword_bounded` a whole-document replay merges across outer pretokens the real path never crosses, so it computes different tokens, not a different time.

### SuperBPE specifics

- `train_bpe` / `train_superbpe` take a `pretokenizer=` scheme name, defaulting to `"gpt2"` so published OWT benchmark numbers stay reproducible. `FileSource`/parquet inputs *reject* a non-default scheme rather than silently ignoring it.
- The original SuperBPE stage-1 regex is the `superbpe_stage1` scheme, not `gpt2`. GPT-2's ` ?\p{L}+` excludes `\p{M}`, so combining marks fall out of the letter run — Devanagari `हिन्दी` becomes 6 pretokens and no consonant+matra unit can ever be learned (measured: −44.75% bytes/token for Hindi at 4k vocab). It also inflates the apparent superword gain, since stage 2 removes boundaries and repairs the damage in the SuperBPE arm only.
- Encoding a trained SuperBPE tokenizer requires exporting `tokenizer.json` with `ByteLevel(use_regex=False)`, which the loader maps to the `superword` scheme (a scheme that yields each segment whole). The released 128k SuperBPE instead ships `Split(\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S), Isolated)`, which maps to `superword_bounded` — an outer scheme that *does* split, and still two-level encodes because `superword_encode_segment` runs level 2 once per outer pretoken. Its `{2,}` (against the o200k family's `+`) is why the level-1 glue rules had to grow: `" ’s"` is one outer piece but two stage-1 pretokens, so the junction is genuinely interior and no amount of outer-awareness would fix it.

## Benchmarks

- `benchmarks/compare/` — cross-library throughput sweep of the *inherited subword* engine. `sweep.py` dedups repos by tokenizer digest and runs one fresh process per measurement; `results.py merge` folds JSONL into `benchmarks/results.json` (best interleaved round, judged by gigatoken throughput) and `results.py render` rewrites the `<!-- benchmarks:start -->` block in `benchmarks/compare/SUBWORD_THROUGHPUT.md`. Never hand-edit that block or `results.json`. It renders to its own document, not the README: the README is about what this fork adds, and it links there instead. Note `render` *appends* the block when the markers are absent, so don't paste the markers into a file you don't want the full matrix in.
- `benchmarks/superbpe/` — the SuperBPE suite (efficiency / throughput / trainer parity / vocabulary differential) writing `results_*.json`, `REPORT.md`, and the README figures in `assets/` (`plot_readme.py`; `superbpe_vs_original.png` is the README's lead figure and needs the reference side of `results_parity.json` plus `results_vocab_diff.json`, else it is skipped). Every script falls back to a synthetic corpus when the OWT file is missing, so it runs offline.
- `pretokenizer_optimization_log.md` and `profiling/` record the step-by-step perf history; consult them before re-litigating a micro-optimization.

## Conventions

- Rust comments are dense and explanatory: they justify *why* an unsafe block is sound (`// SAFETY:` on every one), why a hand-rolled sequence beats the obvious form, and what was measured. Match that density when editing hot code; include the measurement when claiming a speedup.
- `ruff` line-length is 180 with preview formatting; `[lints.rust] unused = "allow"`.
- `benches` inherit release codegen with debug info (profiling parity); the `profiling` profile adds frame pointers.
- Per `CONTRIBUTING.md`, upstream accepts issues rather than PRs; keep diffs concise and scoped.
