# Fast WordPiece (BERT family) Implementation Plan

> **Status: executed.** Every task below is done; the design doc records four
> places where re-running the oracle contradicted the plan's text (HF's
> punctuation set is not ICU's `\p{P}` — 143 codepoints differ; the same for
> `clean_text` — 20; the fill order omitted U+FFFD; the decoder's `cleanup` has
> nine rules, not eleven). Task 9's measurements are in
> `pretokenizer_optimization_log.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Load `google-bert/bert-base-uncased` and family, and encode bit-identically to HuggingFace `tokenizers` at `add_special_tokens=False`.

**Architecture:** WordPiece becomes a third algorithm arm inside the existing `bpe::tiktoken::Tokenizer`, not a new backend struct. The encode hot path (`probe_emit_chunk`) is already model-agnostic — it walks spans, probes the pretoken cache, emits packed lanes — and `encode_pretoken_miss` already dispatches to a second algorithm on an `Option` field (`ranked_merges`). Adding a third arm inherits the pretoken cache and its vocab seeding, `batch.rs`'s worker pool and serial mirrors, the PyO3 surface, padding/truncation, and the added-token matcher unchanged, and needs **zero** Python dispatch changes. Two new pieces sit alongside it: a `bert` pretokenizer scheme (scalar walker) and a materialised `BertNormalizer`.

**Tech Stack:** Rust nightly (pinned by `rust-toolchain.toml`), PyO3/maturin, ICU (`icu::properties`) for the class tables, `icu::normalizer` for NFD, pytest + `tokenizers` 0.22.2 for HF parity, criterion for benches.

## Global Constraints

- Design source of truth: `docs/superpowers/specs/2026-08-10-fast-wordpiece-design.md`. **Read it before starting.** Its "Verified HF semantics" section was measured against `tokenizers` 0.22.2, not recalled — do not "simplify" any of it without re-running the oracle. Several items are counter-intuitive (U+0378 is kept; non-ASCII symbols are not punctuation; strip_accents defaults to the lowercase flag).
- Branch is `wordpiece` (forked from `v3`). Do **not** commit to `main`. Do not push.
- The crate is `gigatoken` (lib `gigatoken_rs`); the fork is a strict superset. Never rename existing public names. All existing gigatoken code must keep working.
- **UTF-8 is trusted, not validated.** Walkers must still be *safe* on invalid UTF-8: never read past `bytes.len()`, never return an end past `len` (that is what `decode_cp` guarantees).
- Every `unsafe` block gets a `// SAFETY:` comment justifying soundness.
- Rust comments are dense and explanatory: justify why a hand-rolled sequence beats the obvious form, and **include the measurement when claiming a speedup**.
- `parallel=False` paths must produce byte-identical output to the parallel ones.
- `ruff` line-length is 180 with preview formatting. Run `uv run ruff check . && uv run ruff format .` before any commit touching Python.
- Rust tests never download. Read the local HF cache via `crate::test_hub::hf_tokenizer_json` and **skip** when the repo is absent.
- New schemes are validated by differential testing against a reference implementation, **not** golden files.
- Scheme naming, fixed: module `bert`, variant `Bert`, name string `"bert"`, struct `FastBertPretokenizer`.
- Out of scope by decision: `TemplateProcessing` / `[CLS]`-`[SEP]` wrapping; a WordPiece trainer; a SIMD `MaskScheme` for the new scheme (Task 2 records the measurement that would justify it); the raw-keyed normalize-on-miss fast path (spec explains the deferral).

---

## File Structure

**Created**

- `src/pretokenize/fast/bert.rs` — scalar walker producing exactly HF `BertPreTokenizer`'s piece sequence, plus its in-file differential test module.
- `src/bpe/wordpiece.rs` — the WordPiece tables and MaxMatch, plus the `BertNormalizer` (or split into `src/bpe/bert_normalizer.rs` if it grows past ~300 lines).

**Modified**

- `src/pretokenize/unicode.rs` — `BertCharClass` + `BertNormClass` packed tables, `is_bert_cjk`, and their ICU-agreement tests.
- `src/pretokenize/fast/mod.rs` — `pub mod` + `pub use` registration.
- `src/pretokenize/options.rs` — enum variant, `pretokenize` arm, `NAMES` (`[&str; 12]` → `[&str; 13]`), `from_name`, `FastPretokenizerDispatch` variant + its `Iterator` and `PretokenSpans` arms, import list.
- `src/pretokenize/mod.rs` — register the scheme in the all-schemes cross-path tests.
- `src/bpe/mod.rs` — `pub mod wordpiece;`.
- `src/bpe/tiktoken.rs` — `wordpiece` and `normalizer` fields, the miss-path arm, `seed_symbols_any` arm, `for_each_piece` normalization hook, `decode` mode, `fork`/`fork_sized` propagation, tests.
- `src/load_tokenizer/hf.rs` — optional `merges`, WordPiece model fields, `ModelProbe` detection, `build_wordpiece`, `parse_bert_normalizer`, `detect_pretokenizer_type` arm, and narrowing `test_unsupported_model_type_named_in_error` to Unigram.
- `tests/tokenizers/conftest.py` — three new `TOKENIZER_SPECS` entries.

---

## Task 1 — Unicode class tables

- [x] `BertCharClass { Word, Whitespace, Punct }` + `BERT_CLASS_TABLE` + `bert_class_of`, following the `CLASS_TABLE`/`ClassTable` pattern (2-bit packed, `LazyLock<Box<[u8]>>`, pre-resolved handle so per-char loops skip the lazy-init check).
- [x] Fill: `\p{P}` → Punct, then ASCII `is_ascii_punctuation` → Punct (adds `$+<=>^`|~`, which are Sc/Sm/Sk), then White_Space → Whitespace.
- [x] `BertNormClass { Keep, Whitespace, Remove, Mark }` + `bert_norm_class_of`. Fill order is load-bearing: White_Space → Whitespace, **then** the control set → Remove, **then** force `\t`/`\n`/`\r` back to Whitespace. Control set is `Cc ∪ Cf ∪ Co ∪ Cs` enumerated explicitly — **not** ICU's `GeneralCategoryGroup::Other`, which includes `Cn` and would wrongly remove U+0378. `Mark` is `Mn` only.
- [x] `is_bert_cjk(cp)` over the 8 BERT ranges (see spec). These differ from `is_deepseek_cjk` — do not reuse it.
- [x] Tests: agree with the ICU predicates over all `0..=char::MAX` scalars (mirror `class_table_matches_icu`), plus explicit pins for U+0378 (Keep), U+0085/U+000B/U+000C (Remove), U+2028/U+2029/U+00A0 (Whitespace), and that `× ÷ € ≠ ☃ ¦ ´ ˈ` and U+0301 are **not** Punct while `¡ « – ' 、 · ་` are.

**Verify:** `cargo test --lib pretokenize::unicode`

## Task 2 — `bert` pretokenizer walker

- [x] `src/pretokenize/fast/bert.rs`, modelled on `superword_bounded.rs`: `FastBertPretokenizer { bytes, pos }`, `new`/`with_pos`/`pos`, `Iterator`, and `unsafe impl PretokenSpans` delegating to `fill_spans_keyed_with_buf`.
- [x] `next_span`: skip whitespace; if the char at `pos` is Punct emit exactly that char; else scan to the next Whitespace-or-Punct char. ASCII is the bulk — use a 128-entry byte table (or SWAR, cf. `swar_scan_letters` in `fast/mod.rs`) before falling back to `decode_cp` + `bert_class_of`.
- [x] Document prominently that **this is the first scheme whose spans do not partition the input** (whitespace is dropped). `fill_spans_keyed_with_buf`'s only contract is `start < end` and in-bounds (`src/pretokenize/mod.rs:452`); `BatchEntry` holds `(ptr, len)` with no contiguity requirement. Every other scheme happens to partition, so a reader will assume it.
- [x] Differential tests vs a reference implementation over `char`s (`is_ascii_punctuation || \p{P}`, White_Space), seeded with the oracle hand cases from the spec, plus randomized codepoint soup drawn from P / S / M / Zs / Cc / Cf / CJK / letters — same shape as `superword_bounded_matches_regex_random`.
- [x] Record in `pretokenizer_optimization_log.md` what the scalar walker measures at, so a future SIMD `MaskScheme` port has a baseline to beat.

**Verify:** `cargo test --lib pretokenize::fast::bert`

## Task 3 — Scheme wiring

- [x] `fast/mod.rs`: `pub mod bert;` + `pub use`.
- [x] `options.rs`: `PretokenizerType::Bert`, `pretokenize` arm, `NAMES` 12 → 13, `from_name("bert")`, `FastPretokenizerDispatch::Bert` + both trait arms.
- [x] **No `from_split_regex` arm** — BERT's pre_tokenizer is `{"type": "BertPreTokenizer"}`, not a `Split`. Detection lands in `detect_pretokenizer_type` (Task 6).
- [x] Register in the all-schemes cross-path tests (`check_all_schemes` / `encode_with_added_tokens_matches_memoized_encode_all_schemes` in `bpe/tiktoken.rs`, and the dispatch-vs-iterator test in `pretokenize/mod.rs`).

**Verify:** `cargo test --lib pretokenize`

## Task 4 — WordPiece model

- [x] `src/bpe/wordpiece.rs` with `head` / `cont` byte-keyed maps (the `##` prefix stripped at load, so the hot loop probes a plain slice with no per-candidate concatenation), `unk`, `max_piece_len`, `max_input_chars_per_word`.
- [x] `encode_unit(&self, bytes, out) -> usize`: greedy longest-match-first; each backoff starts at `min(remaining, max_piece_len)` to bound probes per position; failure anywhere ⇒ emit a single `unk` for the **whole** word.
- [x] Char count for the length cap: `bytes.iter().filter(|b| (b & 0xC0) != 0x80).count()`. It counts **chars, not bytes**.
- [x] Comment why byte-wise backoff is exactly HF: vocab keys are valid UTF-8 strings so no key ends mid-character, and `start` is always a previous match end (a char boundary). This is the reason no char-boundary walking is needed — it is a proof, not an approximation, and a future reader will want it.
- [x] Unit tests from the spec's measured cases: `unaffable`, `abce → UNK`, `abc → UNK`, the 4-char/8-byte length-cap boundary, a custom `continuing_subword_prefix`.

**Verify:** `cargo test --lib bpe::wordpiece`

## Task 5 — `BertNormalizer`

- [x] Struct with the four flags; resolve `strip_accents: Option<bool>` to `unwrap_or(lowercase)` **at construction** so the encode path never re-derives it.
- [x] `normalize<'a>(&self, input: &'a [u8], buf: &'a mut String) -> &'a [u8]` returning the input untouched when nothing applies, mirroring `nfc_segment` (`bpe/tiktoken.rs:109`).
- [x] Order: `clean_text` → `handle_chinese_chars` → `strip_accents` (NFD then drop `Mn`) → `lowercase` (`char::to_lowercase`, may expand).
- [x] ASCII fast path: for pure-ASCII input with no controls, the whole thing collapses to an optional `to_ascii_lowercase`. This is the common case and should not pay per-char table lookups.
- [x] Hook into `for_each_piece` (`bpe/tiktoken.rs:1101`) alongside `normalize_nfc`, with its own scratch buffer.
- [x] Tests: every row of the spec's `clean_text` table, the three `(lowercase, strip_accents)` combinations over the spec's case list, `中文x → " 中  文 x"`, `U+2126 → U+03C9`/`U+03A9`, `U+0130 → i`, `U+0345 → ""`.

**Verify:** `cargo test --lib bpe::wordpiece` (or `bpe::bert_normalizer`)

## Task 6 — Loader

- [x] `Model`: `merges` gains `#[serde(default)]`; add `unk_token`, `continuing_subword_prefix`, `max_input_chars_per_word`.
- [x] `ModelProbe`: WordPiece iff `model.type == "WordPiece"` **or** (untyped) `max_input_chars_per_word.is_some()`. `bert-base-uncased` has **no** `model.type`. Do **not** use `continuing_subword_prefix` — the existing comment at `hf.rs:226` correctly notes BPE serializes it too. Keep the Unigram refusal intact.
- [x] `build_wordpiece` returning `bpe::tiktoken::Tokenizer` (so it rides the existing `HfTokenizer::Bpe` arm and no enum variant is needed). Must **not** call `ByteRemapping::from_byte_vocab` — it errors when the vocab lacks single-byte tokens for every UTF-8-legal byte, which WordPiece vocabs do.
- [x] `parse_bert_normalizer`; `detect_pretokenizer_type` gains a `"BertPreTokenizer"` arm → `PretokenizerType::Bert`.
- [x] Narrow `test_unsupported_model_type_named_in_error` — it currently asserts WordPiece is *rejected by name* for both the typed and untyped-legacy cases, so both WordPiece rows must go and the Unigram rows stay.

**Verify:** `cargo test --lib load_tokenizer`

## Task 7 — `Tokenizer` integration

- [x] Fields `wordpiece: Option<Arc<WordPiece>>` and `normalizer: Option<Arc<BertNormalizer>>` (Arc so `fork`/`fork_sized` share one copy, like `merges`/`vocab`). Propagate through `fork`, `fork_sized`, and `from_tables`.
- [x] `encode_pretoken_miss`: dispatch to `encode_pretoken_miss_wordpiece` **first**, mirroring the `ranked_merges` branch at `:1673` — one perfectly-predicted test that keeps BPE codegen unchanged. It **must** precede any byte remapping: `byte_remapping` is `None` here and the fallback maps byte `b` to `TokenId(b)`, meaningless in a WordPiece vocab.
- [x] `seed_symbols_any` (`:496`) gains a WordPiece arm routing through the same `encode_unit`, so a seeded value can never disagree with a cold miss (the invariant `seeded_pretoken_cache`'s docs establish for BPE).
- [x] `decode`: WordPiece mode — join pieces with spaces, splice out ` ##`, apply `cleanup`.
- [x] `superword` must stay `None` for WordPiece; assert or document the incompatibility.
- [x] Tests: `wordpiece_encode_matches_hf_fixture` on a small committed vocab, `parallel=False` byte-identity, and a `bert-base-uncased` test gated on the local HF cache via `crate::test_hub`.

**Verify:** `cargo test`

## Task 8 — Python parity

- [x] Confirm **no** Python changes are needed (`BPETokenizer` already wraps `Tokenizer`; `gigatoken._tokenizer.Tokenizer` dispatches on backend type). Verify `Tokenizer.from_json` and config dispatch pick it up as-is; only add code if they don't.
- [x] `tests/tokenizers/conftest.py::TOKENIZER_SPECS`: `bert-base-uncased` (lowercase + strip_accents), `bert-base-cased` (`strip_accents=null` with `lowercase=false`), `bert-base-multilingual-cased` (CJK + accents + non-Latin).
- [x] `TokenizerSpec` assumes an end-of-text token; BERT has `[SEP]`/`[PAD]` instead. Either extend the struct or exclude these specs from `test_endoftext_id`.
- [x] Dedicated tests beyond the shared suite: whole-word `[UNK]`, the `max_input_chars_per_word` boundary at 100 chars vs bytes, the `clean_text` table, and CJK/accent cases.

**Verify:**
```bash
uv run pytest tests/tokenizers -k bert -q
uv run pytest tests -q                  # ~1470 tests, must not regress
uv run ruff check . && uv run ruff format . && uv run ty check
cargo bench --bench encode              # confirm no BPE-path regression
```

The real gate is `tests/tokenizers/test_hf_parity.py::test_owt_matches_hf` over a real corpus slice, token-for-token against `tokenizers`.

## Task 9 — Measure and record

- [x] Add a `bert`/WordPiece row to the throughput sweep and record the number. If the materialised normalization pass shows up as more than ~1%, that is the trigger for the deferred raw-keyed fast path — and it needs the differential fuzz described in the spec, not an argument.
- [x] Note the miss-path cost. The naive MaxMatch runs on ~1% of pretokens, so the LinMaxMatch trie (Song et al. 2021) is expected to be under 1% end-to-end; build it only if a bench isolating the miss path says otherwise, and record the measurement either way.
