//! Fast scalar pretokenizer for HF's `BertPreTokenizer` (BERT/WordPiece
//! family). Not a regex scheme: HF applies two successive splits,
//!
//! ```text
//! split(char::is_whitespace, Removed)   then   split(is_bert_punc, Isolated)
//! ```
//!
//! so a piece is either a maximal run of chars that are neither whitespace nor
//! punctuation, or a single punctuation char. `is_bert_punc` is
//! `is_ascii_punctuation(c) || c.is_punctuation()`, which is **not** ICU's
//! `\p{P}` — see the staleness deltas in [`crate::pretokenize::unicode`].
//!
//! # This scheme's spans do not partition the input
//!
//! Every other scheme in this crate emits spans that tile the input end to
//! end, and a reader will assume that here too. This one **drops** whitespace:
//! `"a b"` yields `"a"` and `"b"`, with byte 1 in no span at all. That is
//! sound. [`crate::pretokenize::fill_spans_keyed_with_buf`]'s only contract on
//! `next` is `start < end` and `end <= bytes.len()`, and a `BatchEntry` is a
//! `(ptr, len)` pair with no contiguity requirement — nothing downstream
//! reconstructs the input from the spans, and the encode path only ever hashes
//! and looks up each span's bytes.
//!
//! Consequence worth knowing: unlike the byte-level schemes, decoding a
//! WordPiece token stream cannot reproduce the input, which is why HF ships a
//! `WordPiece` decoder that re-inserts spaces heuristically rather than
//! inverting the split.
//!
//! Scalar, not a [`super::mask::MaskScheme`]: every boundary here is a pure
//! per-byte/per-char class test with no long-skip structure to exploit
//! (`memchr` has nothing to seek — the stop set is dense), and the inner loops
//! are already one table load per byte. A SIMD port is plausible — the class
//! test vectorizes as a shuffle-based table lookup — and the measurement that
//! would justify it is recorded in `pretokenizer_optimization_log.md`.
//!
//! One further fast-path opening, noted here because it is easy to lose: when
//! the tokenizer's `BertNormalizer` has `clean_text` on (all three BERT
//! reference models do), the walker's input can contain **no** whitespace
//! other than U+0020 and no control chars at all — `clean_text` maps every
//! White_Space char to a space and deletes the rest. Measured over every
//! scalar, not argued. The walker does not assume this, because `clean_text`
//! is a config flag.

use super::decode_cp;
use crate::pretokenize::Pretoken;
use crate::pretokenize::unicode::{
    BERT_BYTE_CLASS, BERT_BYTE_NON_ASCII, BERT_BYTE_PUNCT, BERT_BYTE_WORD, BERT_BYTE_WS,
    BertCharClass, BertClassTable,
};

pub struct FastBertPretokenizer<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> FastBertPretokenizer<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Resume iteration at a byte offset previously returned by [`Self::pos`].
    #[inline]
    pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }

    /// Current position as a byte offset into the input.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }
}

impl<'a> Iterator for FastBertPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = next_span(self.bytes, self.pos, BertClassTable::get())?;
        self.pos = end;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

// SAFETY: delegates to `fill_spans_keyed_with_buf`, which writes exactly the
// first `n` entries from live in-bounds spans of `self.bytes`. `next_span`
// returns `start < end <= bytes.len()` and a strictly advancing `end`, so
// every span is nonempty and in bounds. The spans need not be contiguous and
// the helper does not require it (see the module docs).
unsafe impl<'a> crate::pretokenize::PretokenSpans<'a> for FastBertPretokenizer<'a> {
    #[inline(never)]
    fn fill_spans_keyed(
        &mut self,
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        let bytes = self.bytes;
        // Resolve the LazyLock handle once per fill rather than per char.
        let table = BertClassTable::get();
        let mut pos = self.pos;
        let n = crate::pretokenize::fill_spans_keyed_with_buf(
            bytes,
            || {
                let (start, end) = next_span(bytes, pos, table)?;
                pos = end;
                Some((start, end))
            },
            batch,
            prefetch,
        );
        self.pos = pos;
        n
    }
}

/// The next piece at or after `pos`, as `(start, end)`; `None` at end of
/// input (possibly after skipping a trailing whitespace run).
#[inline(always)]
fn next_span(bytes: &[u8], mut pos: usize, table: BertClassTable) -> Option<(usize, usize)> {
    let len = bytes.len();
    // Phase 1: skip the whitespace run, and return early if what follows is a
    // lone punctuation char (`Isolated` makes it its own piece).
    let first_len = loop {
        if pos >= len {
            return None;
        }
        // SAFETY: pos < len checked just above.
        let b = unsafe { *bytes.get_unchecked(pos) };
        match BERT_BYTE_CLASS[b as usize] {
            BERT_BYTE_WS => pos += 1,
            BERT_BYTE_PUNCT => return Some((pos, pos + 1)),
            BERT_BYTE_WORD => break 1,
            // Non-ASCII: decode once and dispatch on the codepoint's class.
            // SAFETY: pos < len and bytes[pos] >= 0x80 — decode_cp's
            // documented precondition; it never reads or returns past `len`,
            // and clamps garbage to a valid scalar so the table index is safe.
            _ => {
                let (cp, l) = unsafe { decode_cp(bytes, pos) };
                match table.class_of(cp) {
                    BertCharClass::Whitespace => pos += l,
                    BertCharClass::Punct => return Some((pos, pos + l)),
                    BertCharClass::Word => break l,
                }
            }
        }
    };

    // Phase 2: the word run — everything up to the next whitespace or
    // punctuation char. ASCII advances one table load per byte; a non-ASCII
    // byte costs one decode.
    let start = pos;
    let mut p = pos + first_len;
    loop {
        // SAFETY (both `get_unchecked`s): guarded by `p < len`.
        while p < len {
            let b = unsafe { *bytes.get_unchecked(p) };
            if BERT_BYTE_CLASS[b as usize] != BERT_BYTE_WORD {
                break;
            }
            p += 1;
        }
        if p >= len {
            return Some((start, len));
        }
        let b = unsafe { *bytes.get_unchecked(p) };
        if BERT_BYTE_CLASS[b as usize] != BERT_BYTE_NON_ASCII {
            // Whitespace or punctuation: the run ends here.
            return Some((start, p));
        }
        // SAFETY: as in phase 1.
        let (cp, l) = unsafe { decode_cp(bytes, p) };
        if table.class_of(cp) != BertCharClass::Word {
            return Some((start, p));
        }
        p += l;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::pretokenize::unicode::bert_class_of;

    /// Reference implementation: HF's two splits, expressed directly over
    /// `char`s. This isolates what a byte walker gets wrong — UTF-8 decoding,
    /// span boundaries, dropping whitespace, isolating punctuation — by
    /// sharing the class data with the walker on purpose. The class *data* is
    /// validated separately, against ICU plus the measured staleness deltas in
    /// `unicode.rs` and against `tokenizers` itself in the Python suite.
    pub(crate) fn reference_tokens(s: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut cur = String::new();
        for c in s.chars() {
            match bert_class_of(c as u32) {
                BertCharClass::Whitespace => {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                }
                BertCharClass::Punct => {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                    out.push(c.to_string());
                }
                BertCharClass::Word => cur.push(c),
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    fn fast_tokens(s: &str) -> Vec<String> {
        FastBertPretokenizer::new(s.as_bytes())
            .map(|t| String::from_utf8_lossy(t.0).into_owned())
            .collect()
    }

    /// Measured against `tokenizers` 0.22.2 (`BertPreTokenizer`), not reasoned
    /// about. Asserted against both the walker and the reference, so the
    /// reference the random test leans on is itself pinned.
    const GROUND_TRUTH: &[(&str, &[&str])] = &[
        ("hello world", &["hello", "world"]),
        ("hi!", &["hi", "!"]),
        ("don't", &["don", "'", "t"]),
        ("a  b", &["a", "b"]),
        ("  lead", &["lead"]),
        ("trail  ", &["trail"]),
        ("   ", &[]),
        ("", &[]),
        ("unaffable", &["unaffable"]),
        // Non-ASCII symbols are word chars to HF, so they fuse with letters.
        ("2×3", &["2×3"]),
        ("a€b", &["a€b"]),
        ("x≠y", &["x≠y"]),
        // Non-ASCII \p{P} is punctuation and isolates.
        ("¡hola!", &["¡", "hola", "!"]),
        ("«x»", &["«", "x", "»"]),
        ("a–b", &["a", "–", "b"]),
        ("中、文", &["中", "、", "文"]),
        // NBSP and U+2003 are whitespace; ZWSP is not (it is Cf, a word char
        // here — clean_text removes it earlier in the real pipeline).
        ("a\u{a0}b", &["a", "b"]),
        ("a\u{2003}b", &["a", "b"]),
        ("a\u{200b}b", &["a\u{200b}b"]),
        // U+001F is a control but NOT White_Space, so it joins the word run.
        ("a\u{1f}b", &["a\u{1f}b"]),
        // Combining marks are word chars.
        ("e\u{301}f", &["e\u{301}f"]),
        // The staleness deltas, in situ.
        ("a\u{9fd}b", &["a\u{9fd}b"]),
        ("a\u{166d}b", &["a", "\u{166d}", "b"]),
        ("a\u{111c9}b", &["a", "\u{111c9}", "b"]),
        // Runs of punctuation isolate one char at a time.
        ("a...b", &["a", ".", ".", ".", "b"]),
        ("!!!", &["!", "!", "!"]),
        // CJK is not special to the pretokenizer (the normalizer's
        // handle_chinese_chars is what separates it, by inserting spaces).
        ("中文x", &["中文x"]),
        (" 中  文 x", &["中", "文", "x"]),
    ];

    #[test]
    fn bert_matches_ground_truth() {
        for (input, want) in GROUND_TRUTH {
            assert_eq!(reference_tokens(input), *want, "reference disagrees on {input:?}");
            assert_eq!(fast_tokens(input), *want, "walker disagrees on {input:?}");
        }
    }

    /// Random codepoint soup over the classes that matter, including the
    /// categories HF's punctuation predicate treats surprisingly (S, M, Cf)
    /// and multi-byte lengths 1–4.
    #[test]
    fn bert_matches_reference_random() {
        use rand::prelude::*;
        let pools: &[&[char]] = &[
            &['a', 'z', 'A', 'Z', 'é', 'ß', 'ж', 'ا', '한', '日', '𐐀'], // letters, 1–4 bytes
            &['0', '9', '٢', '½', 'Ⅷ', '๕', '１'],                      // numbers
            &[' ', '\t', '\n', '\r', '\u{b}', '\u{c}', '\u{85}', '\u{a0}', '\u{2028}', '\u{2003}', '\u{3000}'], // White_Space
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}', '\u{111c9}'], // marks (one is HF-punct)
            &['.', ',', '!', '$', '\'', '+', '<', '`', '~', '¡', '«', '»', '–', '、', '·'], // punct
            &['×', '÷', '€', '≠', '☃', '¦', '´', 'ˈ', '\u{166d}'],       // symbols (one is HF-punct)
            &['\u{0}', '\u{7}', '\u{1f}', '\u{7f}', '\u{ad}', '\u{200b}', '\u{feff}', '\u{378}', '\u{e000}'], // C*
            &['中', '文', '\u{3400}', '\u{2b820}', '\u{f900}'],          // CJK incl. the range hole
            &['\u{9fd}', '\u{c77}', '\u{2e4f}', '\u{11660}'],            // stale-not-punct deltas
        ];
        let mut rng = StdRng::seed_from_u64(0xBE27_C1A5);
        for round in 0..4000 {
            let len = rng.random_range(1..40);
            let s: String = (0..len)
                .map(|_| {
                    let pool = pools.choose(&mut rng).unwrap();
                    *pool.choose(&mut rng).unwrap()
                })
                .collect();
            assert_eq!(
                fast_tokens(&s),
                reference_tokens(&s),
                "Mismatch on round {round}, case {s:?}"
            );
        }
    }

    /// Invalid UTF-8 must not panic or produce an out-of-bounds span. UTF-8 is
    /// trusted by contract, but a walker still has to stay *safe* on garbage:
    /// truncated sequences at the buffer end are the classic way an unchecked
    /// decode walks off the slice.
    #[test]
    fn bert_safe_on_invalid_utf8() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(0x1BAD_5EED);
        for _ in 0..2000 {
            let len = rng.random_range(1..48);
            let bytes: Vec<u8> = (0..len).map(|_| rng.random::<u8>()).collect();
            let mut last_end = 0usize;
            for span in FastBertPretokenizer::new(&bytes) {
                let start = span.0.as_ptr() as usize - bytes.as_ptr() as usize;
                let end = start + span.0.len();
                assert!(!span.0.is_empty(), "empty span on {bytes:?}");
                assert!(end <= bytes.len(), "span past end on {bytes:?}");
                assert!(start >= last_end, "spans went backwards on {bytes:?}");
                last_end = end;
            }
        }
    }

    /// `with_pos` must resume exactly where iteration left off, which is what
    /// the chunked batch path relies on.
    #[test]
    fn bert_with_pos_resumes() {
        let s = "Hello, wörld! 中文 x";
        let all = fast_tokens(s);
        let mut it = FastBertPretokenizer::new(s.as_bytes());
        let first = it.next().unwrap();
        let resumed: Vec<String> = FastBertPretokenizer::with_pos(s.as_bytes(), it.pos())
            .map(|t| String::from_utf8_lossy(t.0).into_owned())
            .collect();
        assert_eq!(String::from_utf8_lossy(first.0), all[0]);
        assert_eq!(resumed, all[1..]);
    }
}
