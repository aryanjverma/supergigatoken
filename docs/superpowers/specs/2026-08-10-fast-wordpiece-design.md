# Fast WordPiece (BERT family) — design

Status: **designed, not implemented.** Branch `wordpiece`, forked from `v3`.

## Goal

Load and fast-encode WordPiece tokenizers (`bert-base-uncased` and family) with
output bit-identical to HuggingFace `tokenizers` at `add_special_tokens=False`,
reusing the existing encode engine rather than standing up a third backend.

Scope decisions taken (defaults; revisit only if a consumer needs otherwise):

- **No post-processor.** No `TemplateProcessing`, so no `[CLS]`/`[SEP]`
  wrapping. This matches every existing tokenizer in this repo — the BPE path
  ignores post-processors too, which is why the parity suite uses
  `add_special_tokens=False`.
- **No trainer.** Encoding only. WordPiece training optimises likelihood gain,
  shares nothing with `bpe_train.rs`, and would roughly double the work.
- **Normalization is materialised**, not fused into the pretokenizer (see
  "Normalization" below).
- **Naive MaxMatch** first, LinMaxMatch trie only behind a measurement.

## Why this is a small change

The encode hot path is already model-agnostic:

```
encode_with_added_tokens_flat        src/bpe/tiktoken.rs:1162
  └─ for_each_piece                                    :1101   added tokens, NFC, prefix space
       └─ memoized_encode_flat                         :1421
            ├─ fill_spans_keyed  (SIMD pretokenize + key + hash + prefetch)
            └─ probe_emit_chunk                        :1457   branchless cache probe/emit
                 └─ probe_emit_slow                    :1548   #[cold]
                      └─ encode_pretoken_miss          :1661   ← the ONLY BPE-specific code
```

`probe_emit_chunk` walks spans, probes `ShortPretokenCache` on the span bytes,
and emits packed token lanes. Nothing in it knows what algorithm produced the
tokens. `encode_pretoken_miss` **already dispatches to a second algorithm** on
an `Option` field:

```rust
if self.ranked_merges.is_some() {
    return self.encode_pretoken_miss_ranked(bytes, key, h, slot, out);
}
```

WordPiece becomes a third arm of exactly that shape. Everything downstream is
inherited unchanged: the pretoken cache and its vocab seeding, `batch.rs`'s
worker pool and serial mirrors, the PyO3 surface, padding/truncation, the
added-token Aho-Corasick matcher, `encode_files`.

The alternative — a `WordPieceTokenizer` struct alongside `SentencePieceBPE` —
would need a **third** parallel family of functions in `batch.rs` (there are
already two: `encode_docs_ragged*` for `Tokenizer` and `sp_encode_docs_ragged*`
for SentencePiece) plus a third pyclass. Rejected.

Consequence: the Python side needs no dispatch change at all. `BPETokenizer`
already wraps `Tokenizer`, and `gigatoken._tokenizer.Tokenizer` selects on
backend type. Only `decode` needs a mode flag.

## Verified HF semantics

**Everything in this section was measured against `tokenizers` 0.22.2, not
recalled.** Reproduce with the probes described at the end. Do not "simplify"
any of it without re-running the oracle — several items are counter-intuitive.

### `BertPreTokenizer`

Two passes: split on whitespace with the delimiter **removed**, then split on
punctuation **isolated**. Empty pieces are dropped. So a pretoken is either a
maximal run of non-whitespace non-punctuation chars, or a single punctuation
char. Whitespace never appears in output.

- **Punctuation set = Rust `is_ascii_punctuation` ∪ `\p{P}`.** Measured
  isolated: ``!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`` and `¡ « – ' 、 · ་`.
  Measured **not** isolated: `× ÷ € ≠ ☃ ¦ ´ ˈ` and combining U+0301.
  So the ASCII set includes `$+<=>^`|~` (which are `Sc`/`Sm`/`Sk`, not `P`),
  but non-ASCII symbols are **not** punctuation. Neither `\p{S}` nor `\p{M}`
  qualifies outside ASCII — this is why the existing `DsCharClass` cannot be
  reused: it lumps `\p{P}` and `\p{S}` into one class.
- **Whitespace = Unicode White_Space.** `U+00A0` and `U+2003` split;
  `U+200B` (ZWSP, `Cf`) does not.

### `BertNormalizer`

Applied in this order, and the order is load-bearing:

1. `clean_text`
2. `handle_chinese_chars`
3. `strip_accents` — **defaults to the value of `lowercase`** when the JSON
   says `null`. Measured: `lowercase=true, strip_accents=null` gives
   `Café → cafe`; `lowercase=false, strip_accents=null` gives `Café → Café`.
4. `lowercase`

**`clean_text`**: drop `\0`, `U+FFFD`, and control chars; then map every
remaining White_Space char to a plain space. Control **wins over** whitespace:

| codepoint | GC | result |
|---|---|---|
| U+0009 / U+000A / U+000D | Cc | → space (explicit exception) |
| U+000B / U+000C / U+001F / U+007F / U+0085 | Cc | **removed** |
| U+00A0 / U+2003 / U+3000 | Zs | → space |
| U+2028 / U+2029 | Zl / Zp | → space |
| U+200B / U+180E | Cf | **removed** |
| U+E000 | Co | **removed** |
| **U+0378** | **Cn (unassigned)** | **KEPT** |

The last row is the trap. HF's control predicate is `Cc ∪ Cf ∪ Co ∪ Cs`
**minus** `\t\n\r`. ICU's `GeneralCategoryGroup::Other` *includes* `Cn`, so
filling the table from that group directly removes U+0378 and diverges. The
four categories must be enumerated explicitly. Pin U+0378 in a test.

Table fill order must therefore be: White_Space first, then the control set
(so `U+0085`, which is both, ends up removed), then force `\t\n\r` back to
whitespace.

**`handle_chinese_chars`**: each CJK char is replaced by `' ' + c + ' '`.
Measured: `中文x → " 中  文 x"` (adjacent CJK yields a double space). Ranges are
the classic BERT set:

```
4E00..=9FFF, 3400..=4DBF, F900..=FAFF,
20000..=2A6DF, 2A700..=2B73F, 2B740..=2B81F, 2B920..=2CEAF, 2F800..=2FA1F
```

Note these are **not** `is_deepseek_cjk`'s ranges (which cover kana and stop at
9FA5); a separate predicate is required.

**`strip_accents`**: NFD, then drop chars with GC `Mn`. It really is canonical
decomposition, not just mark removal — measured `U+2126 OHM SIGN → U+03C9`
(with lowercase) and `→ U+03A9` (strip only). Compatibility decompositions are
**not** applied: `U+01C5 → U+01C6` comes from `to_lowercase`, not NFD.

**`lowercase`**: per-char `char::to_lowercase`, which may expand. Interaction
with strip_accents, measured: `U+0130 İ` → NFD `I` + `U+0307` → strip → `I` →
lower → `i`. A string that is entirely marks can normalise to empty
(`U+0345 → ""`).

### WordPiece model

`bert-base-uncased`'s `model` block has **no `"type"` field** — keys are
exactly `{unk_token, continuing_subword_prefix, max_input_chars_per_word,
vocab}`. Vocab 30522 entries, 5828 with `##`, longest piece 18 bytes.
Detection must handle the untyped-legacy case.

Algorithm, per pretoken, all measured:

- Greedy longest-match-first from the left; non-initial pieces are looked up
  with `continuing_subword_prefix` prepended. `unaffable → un ##aff ##able`.
- **Failure anywhere ⇒ the whole word becomes one `[UNK]`**, not a partial
  emission. Measured with vocab `{ab, ##cd}`: `abcd → [ab, ##cd]` but
  `abce → [UNK]` and `abc → [UNK]`.
- `max_input_chars_per_word` (default 100) counts **chars, not bytes**.
  Measured: four `é` (8 bytes, 4 chars) passes at limit 4, fails at limit 3.
- The backoff decrements the **end offset**, and this can be done **byte-wise**
  with no char-boundary walking: vocab keys are valid UTF-8 strings, so no key
  can ever end mid-character, and `start` is always a previous successful match
  end (hence a char boundary). A byte-slice probe against a byte-keyed map is
  therefore *exactly* HF's result. This is a genuine simplification, not an
  approximation.

### Decoder

`{"type": "WordPiece", "prefix": "##", "cleanup": true}` — join pieces with
spaces, then splice out ` ##`. `cleanup` additionally tidies spacing around
punctuation and contractions. `Tokenizer::decode` currently just concatenates
vocab bytes, so it needs a WordPiece mode.

## Normalization strategy

HF's order is `normalize(whole text) → pretokenize → model`, but gigatoken's
speed comes from a cache keyed on pretoken bytes, and `BertNormalizer` rewrites
bytes *before* the split.

**Chosen: materialise.** Normalize each added-token segment into a scratch
buffer inside `for_each_piece`, then pretokenize and encode the normalized
bytes. Exactly HF's semantics, no edge cases, and `handle_chinese_chars` falls
out for free (the inserted spaces are dropped by the whitespace split, so CJK
chars become their own pretokens without any pretokenizer special-casing). The
SentencePiece backend already does this — `encode_normalized_cb`,
`src/bpe/sentencepiece.rs:1154` — so the shape is precedented.

Cost is one linear pass. ASCII lowercase is a few GB/s against an encode path
running at hundreds of MB/s, so this should be well under a percent. Measure
before optimising.

**Deferred: raw-keyed, normalize-on-miss.** Key the cache on raw input bytes,
normalize each word only on the ~1% miss path, fold `handle_chinese_chars` into
the pretokenizer as a third isolated class. Zero extra passes and a better hit
rate. The reason it is deferred rather than chosen: it needs the claim "the
split points come out the same" to hold, and while each individual case appears
to work out (control chars removed inside a word cannot join two words that the
raw splitter had already joined; NFD and `to_lowercase` are class-preserving; a
word normalising to empty just emits nothing), that is an argument, not a proof.
This repo's convention is to establish such things by differential fuzz against
the materialised path. Do that only if measurement says the pass matters.

## Implementation

### 1. `src/pretokenize/unicode.rs` — two class tables

Follow the existing `CLASS_TABLE` / `ClassTable` pattern (2-bit packed,
`LazyLock<Box<[u8]>>`, pre-resolved handle to avoid per-call lazy-init checks).

- `BertCharClass { Word, Whitespace, Punct }` + `bert_class_of` — the
  pretokenizer's hot path. Fill `\p{P}`, then ASCII punctuation, then
  White_Space (all three sets are pairwise disjoint except that ASCII punct
  overlaps `\p{P}`, harmlessly).
- `BertNormClass { Keep, Whitespace, Remove, Mark }` + `bert_norm_class_of` —
  one load serves both `clean_text` and `strip_accents`. Fill order per the
  table above. `Mark` is `Mn` only.
- `is_bert_cjk(cp)` for `handle_chinese_chars`.

Tests: agree with ICU predicates over all 0x110000 scalars (mirroring
`class_table_matches_icu`), plus explicit pins for U+0378, U+0085, U+000B,
U+2028, and the `× ÷ € ≠ ¦ ´ ˈ`-are-not-punctuation set.

### 2. `src/pretokenize/fast/bert.rs` — the walker

Scalar, modelled on `superword_bounded.rs` (which is the existing scalar-walker
template and documents why a scalar walker can be the right call). Structure:
`FastBertPretokenizer { bytes, pos }`, `new`/`with_pos`/`pos`, `Iterator`, and
`unsafe impl PretokenSpans` delegating to `fill_spans_keyed_with_buf`.

`next_span`: skip whitespace; if the char at `pos` is punctuation emit just it;
else scan to the next whitespace-or-punctuation char. ASCII is the bulk, so the
inner scan should use a 128-entry byte table before falling back to `decode_cp`
+ `bert_class_of` (see `swar_scan_letters` in `fast/mod.rs` for the SWAR idiom).

**This is the first scheme that drops input bytes.** That is fine —
`fill_spans_keyed_with_buf`'s only contract is `start < end` and in-bounds
(`src/pretokenize/mod.rs:452`), and `BatchEntry` holds `(ptr, len)` pairs with
no contiguity requirement. Nothing assumes the spans partition the input.
Worth a comment saying so, since every other scheme does partition it.

A SIMD `MaskScheme` version is possible later (boundaries are a pure per-byte
class test, well suited to `mask.rs`), but measure the scalar walker first.

Differential tests against a reference implementation built on `\p{P}` +
`is_ascii_punctuation` + White_Space, over hand cases and randomized codepoint
soup drawn from the classes that matter (P, S, M, Zs, Cc, Cf, CJK, letters) —
same shape as `superword_bounded_matches_regex_random`. Seed the hand cases
from the oracle table above.

### 3. `src/pretokenize/options.rs` — wiring

Per the "Adding a pretokenizer scheme" checklist in CLAUDE.md: enum variant
`Bert`, `pretokenize` dispatch arm, `NAMES` (bump 12 → 13), `from_name`,
`FastPretokenizerDispatch` variant + both trait impls. `from_split_regex` does
**not** apply — BERT's pre_tokenizer is `{"type": "BertPreTokenizer"}`, not a
`Split`, so detection happens in `detect_pretokenizer_type`
(`src/load_tokenizer/hf.rs:603`) by kind name instead.

Register the scheme in the existing all-schemes cross-path tests
(`encode_with_added_tokens_matches_memoized_encode_all_schemes`,
`check_all_schemes`).

### 4. `src/bpe/wordpiece.rs` — the model

```rust
pub struct WordPiece {
    head: HashMap<Box<[u8]>, TokenId, FxBuildHasher>,   // no prefix
    cont: HashMap<Box<[u8]>, TokenId, FxBuildHasher>,   // `##` stripped
    unk: TokenId,
    max_piece_len: usize,          // longest key in either map, caps the backoff
    max_input_chars_per_word: usize,
}
```

Splitting the prefix off at load time means the hot loop probes a plain byte
slice with no string concatenation per candidate. `encode_unit(&self, bytes,
out) -> usize` implements the measured algorithm; char count for the length cap
is `bytes.iter().filter(|b| (b & 0xC0) != 0x80).count()`.

Cap each backoff at `max_piece_len` (18 bytes for bert-base-uncased) rather
than starting from the word end — bounds the probe count per position.

### 5. `Tokenizer` integration — `src/bpe/tiktoken.rs`

- New field `wordpiece: Option<Arc<WordPiece>>` (Arc so `fork`/`fork_sized`
  share one copy, like `merges`/`vocab`).
- `encode_pretoken_miss`: dispatch to `encode_pretoken_miss_wordpiece` first,
  mirroring the `ranked_merges` branch at `:1673` — one perfectly-predicted
  test, keeping BPE codegen unchanged. **Must dispatch before any byte
  remapping**: `byte_remapping` is `None` for WordPiece and the fallback maps
  byte `b` to `TokenId(b)`, which is meaningless in a WordPiece vocab.
- `seed_symbols_any` (`:496`) gets a WordPiece arm so the vocab-seeded cache
  agrees with a cold miss. Note the seed values are *not* trivially each
  entry's own ID: `##ing` is not a whole word, and a whole word in the vocab
  does MaxMatch to its own single ID, so routing seeding through the same
  `encode_unit` the miss path uses is both correct and simplest.
- `ByteRemapping::from_byte_vocab` must **not** be called — it errors when a
  vocab lacks single-byte tokens for every UTF-8-legal byte, which WordPiece
  vocabs do.
- `decode`: WordPiece mode (join with spaces, splice ` ##`, then `cleanup`).
- A `normalizer: Option<Arc<BertNormalizer>>` field consumed in
  `for_each_piece` (`:1101`) next to `normalize_nfc`.
- `superword` must stay `None`; assert or document the incompatibility.

### 6. `src/load_tokenizer/hf.rs`

- `Model.merges` becomes `#[serde(default)]` (WordPiece has none), plus
  `unk_token`, `continuing_subword_prefix`, `max_input_chars_per_word`.
- `ModelProbe`: detect WordPiece by `model.type == "WordPiece"` **or**, for
  untyped files, `max_input_chars_per_word.is_some()`. The existing comment at
  `:226` correctly warns that `continuing_subword_prefix` is not a valid
  discriminator — BPE serializes it too. Keep the Unigram refusal untouched.
- `HfTokenizer` gains no variant: `build_wordpiece` returns a
  `bpe::tiktoken::Tokenizer`, so it rides the existing `HfTokenizer::Bpe` arm
  and `load_hf_slice`'s dispatch needs only the model-family branch.
- Replace the refusal in `parse_tokenizer_json` (`:235`) with the build path,
  and update `test_unsupported_model_type_named_in_error` — it currently
  asserts WordPiece is *rejected* by name, so it must be narrowed to Unigram.
- `parse_bert_normalizer` for the four flags; `detect_pretokenizer_type` grows
  a `"BertPreTokenizer"` arm.

### 7. Python + tests

Python needs no changes (see "Why this is a small change"). Verify
`Tokenizer.from_json` and config dispatch pick it up as-is.

Register in `tests/tokenizers/conftest.py::TOKENIZER_SPECS` so the whole
existing parity suite (encode, added tokens, decode roundtrip, batch==single,
OWT) covers them:

| spec | why |
|---|---|
| `bert-base-uncased` | lowercase + strip_accents, the reference case |
| `bert-base-cased` | `strip_accents=null` with `lowercase=false` — the no-strip default |
| `bert-base-multilingual-cased` | CJK, accents, non-Latin scripts |

Note `eot_text`/`eot_id` in `TokenizerSpec` assume an end-of-text token; BERT
has `[SEP]`/`[PAD]` instead, so either extend the spec struct or exclude these
from `test_endoftext_id`.

Dedicated tests beyond the shared suite: whole-word `[UNK]`,
`max_input_chars_per_word` boundary at 100 chars vs bytes, the `clean_text`
table above, and `parallel=False` byte-identity.

## Verification

```bash
cargo test --lib pretokenize::fast::bert
cargo test --lib pretokenize::unicode
cargo test --lib bpe::wordpiece
cargo test
uv run pytest tests/tokenizers -k bert -q
uv run pytest tests/test_load_hf.py tests/test_encode.py -q
uv run pytest tests                     # full suite, ~1470 tests, must not regress
uv run ruff check . && uv run ruff format . && uv run ty check
cargo bench --bench encode              # confirm no BPE-path regression
```

Parity is the real gate: `tests/tokenizers/test_hf_parity.py::test_owt_matches_hf`
over a real corpus slice, token-for-token against `tokenizers`.

## Reproducing the oracle

The probes that produced the tables above (throwaway, not committed):

- `BertPreTokenizer`: for each candidate char `c`, check whether
  `pre_tokenize_str("a" + c + "b")` yields `["a", c, "b"]`.
- `BertNormalizer`: `normalize_str` over the case list, with the three
  `(lowercase, strip_accents)` combinations, comparing `ascii()` reprs.
- `clean_text` precedence: `normalize_str("a" + chr(cp) + "b")` and classify
  the result as space / removed / kept.
- WordPiece: build `Tokenizer(WordPiece(vocab, unk_token="[UNK]", ...))` over
  small hand vocabs and read back `encode(s, add_special_tokens=False).tokens`.

On Windows set `PYTHONIOENCODING=utf-8` and print `ascii()` reprs — the console
is cp1252 and will otherwise raise on the non-Latin cases.
