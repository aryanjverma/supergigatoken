//! WordPiece (BERT family): greedy longest-match-first segmentation against a
//! vocabulary, with a whole-word `[UNK]` on any failure.
//!
//! This is a *model*, not a backend: it plugs into `bpe::tiktoken::Tokenizer`'s
//! pretoken-miss path, so it inherits the pretoken cache, the batch worker
//! pool, the PyO3 surface, and padding/truncation unchanged. See
//! `docs/superpowers/specs/2026-08-10-fast-wordpiece-design.md`.
//!
//! Every behavioural claim below was measured against `tokenizers` 0.22.2.

use crate::token::TokenId;
use eyre::{Result, eyre};
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

/// HF's default when `model.max_input_chars_per_word` is absent.
pub const DEFAULT_MAX_INPUT_CHARS_PER_WORD: usize = 100;

/// tokenizer.json's `decoder` block when it is
/// `{"type": "WordPiece", "prefix": "##", "cleanup": true}`.
///
/// Both fields matter and neither can be assumed. Measured on `tokenizers`
/// 0.22.2 with the vocab `{ab, ##cd, hello, .}` decoding `[ab, ##cd, hello, .]`:
///
/// | decoder | result |
/// |---|---|
/// | absent | `"ab ##cd hello ."` |
/// | `cleanup: true` | `"abcd hello."` |
/// | `cleanup: false` | `"abcd hello ."` |
///
/// So the space join is what HF does with *no* decoder at all; this block adds
/// the ` <prefix>` splice, and `cleanup` adds the punctuation tidy on top. A
/// decoder-less WordPiece file must therefore keep its `##` markers and its
/// space before the period — which is why this is `Option` on the tokenizer
/// rather than a pair of defaults.
pub struct WordPieceDecoder {
    pub prefix: Box<str>,
    pub cleanup: bool,
}

pub struct WordPiece {
    /// Pieces that may start a word (no continuing prefix).
    head: HashMap<Box<[u8]>, TokenId, FxBuildHasher>,
    /// Pieces that continue a word, keyed with `continuing_subword_prefix`
    /// **stripped**. Splitting the prefix off at load time is what lets the hot
    /// loop probe a plain byte slice: HF re-formats `"##" + substr` on every
    /// candidate, which is an allocation per probe.
    cont: HashMap<Box<[u8]>, TokenId, FxBuildHasher>,
    unk: TokenId,
    /// Longest key in `head` / longest *stripped* key in `cont`. The backoff
    /// starts here rather than at the word end, which bounds probes per
    /// position at `max_*_len` instead of the word length. Sound because a
    /// candidate longer than every key in the map it is probed against cannot
    /// match, so HF's extra iterations all fail.
    max_head_len: usize,
    max_cont_len: usize,
    /// HF's cap, counted in **chars, not bytes** (measured: a 4-char, 8-byte
    /// word passes at a limit of 4 and fails at 3).
    max_input_chars_per_word: usize,
    /// Kept for `decode`, which splices ` <prefix>` back out.
    prefix: Box<str>,
}

impl std::fmt::Debug for WordPiece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WordPiece {{ head: {}, cont: {}, unk: {:?}, max_input_chars_per_word: {} }}",
            self.head.len(),
            self.cont.len(),
            self.unk,
            self.max_input_chars_per_word
        )
    }
}

impl WordPiece {
    /// Build from a `tokenizer.json` WordPiece model's vocab.
    pub fn new(
        vocab: impl IntoIterator<Item = (String, TokenId)>,
        unk_token: &str,
        continuing_subword_prefix: &str,
        max_input_chars_per_word: usize,
    ) -> Result<Self> {
        let mut head: HashMap<Box<[u8]>, TokenId, FxBuildHasher> =
            HashMap::with_hasher(FxBuildHasher);
        let mut cont: HashMap<Box<[u8]>, TokenId, FxBuildHasher> =
            HashMap::with_hasher(FxBuildHasher);
        let mut unk = None;
        let mut max_head_len = 0usize;
        let mut max_cont_len = 0usize;
        for (piece, id) in vocab {
            if piece == unk_token {
                unk = Some(id);
            }
            // An empty prefix (HF allows it) would make every piece a `cont`
            // piece *and* a `head` piece; `strip_prefix` on "" succeeds for
            // every string, so guard it rather than silently emptying `head`.
            match continuing_subword_prefix.is_empty() {
                true => {
                    max_head_len = max_head_len.max(piece.len());
                    max_cont_len = max_cont_len.max(piece.len());
                    head.insert(piece.as_bytes().into(), id);
                    cont.insert(piece.as_bytes().into(), id);
                }
                false => match piece.strip_prefix(continuing_subword_prefix) {
                    Some(rest) => {
                        // `##` alone is a legal vocab entry (bert-base-uncased
                        // has none, but nothing forbids it); it strips to the
                        // empty string, which the walk never probes.
                        max_cont_len = max_cont_len.max(rest.len());
                        cont.insert(rest.as_bytes().into(), id);
                    }
                    None => {
                        max_head_len = max_head_len.max(piece.len());
                        head.insert(piece.as_bytes().into(), id);
                    }
                },
            }
        }
        let unk = unk.ok_or_else(|| {
            eyre!("WordPiece vocab does not contain its unk_token {unk_token:?}")
        })?;
        Ok(WordPiece {
            head,
            cont,
            unk,
            max_head_len,
            max_cont_len,
            max_input_chars_per_word,
            prefix: continuing_subword_prefix.into(),
        })
    }

    /// The `continuing_subword_prefix` (`"##"` for BERT), for the decoder.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    pub fn unk_id(&self) -> TokenId {
        self.unk
    }

    /// Segment one pretoken, appending its token IDs to `out`. Returns how
    /// many were appended.
    ///
    /// # Why a byte-wise backoff is *exactly* HF, not an approximation
    ///
    /// HF decrements the end offset one byte at a time and probes the
    /// substring, which for a `&str` means most candidate ends are not char
    /// boundaries. Those can never match: every vocab key is a valid UTF-8
    /// string, so no key ends mid-character, and `start` is always either 0 or
    /// a previous successful match end (hence a char boundary). Probing a
    /// non-boundary byte slice against a byte-keyed map therefore fails, which
    /// is indistinguishable from skipping it. No char-boundary walking is
    /// needed anywhere in this function.
    pub fn encode_unit(&self, bytes: &[u8], out: &mut Vec<TokenId>) -> usize {
        if bytes.is_empty() {
            // HF's loop body never runs, `is_bad` stays false, and it emits
            // nothing — notably not an `[UNK]`. Reachable in the real pipeline:
            // a word consisting only of combining marks normalizes to empty
            // under `strip_accents` (measured: `"ab ͅ ab"` → `[ab, ab]`).
            return 0;
        }
        // Char count, without decoding: every non-continuation byte starts one.
        let char_len = bytes.iter().filter(|&&b| (b & 0xC0) != 0x80).count();
        if char_len > self.max_input_chars_per_word {
            out.push(self.unk);
            return 1;
        }

        let mark = out.len();
        let len = bytes.len();
        let mut start = 0usize;
        while start < len {
            let (map, cap) = if start == 0 {
                (&self.head, self.max_head_len)
            } else {
                (&self.cont, self.max_cont_len)
            };
            let mut end = len.min(start + cap);
            let matched = loop {
                if start >= end {
                    break None;
                }
                // SAFETY-free: plain slice index, start < end <= len.
                if let Some(&id) = map.get(&bytes[start..end]) {
                    break Some((id, end));
                }
                end -= 1;
            };
            match matched {
                Some((id, end)) => {
                    out.push(id);
                    start = end;
                }
                None => {
                    // Failure anywhere makes the **whole word** one `[UNK]`,
                    // not a partial emission: measured with vocab
                    // `{ab, ##cd}`, `"abcd"` → `[ab, ##cd]` but `"abc"` →
                    // `[UNK]` and `"abcdab"` → `[UNK]`.
                    out.truncate(mark);
                    out.push(self.unk);
                    return 1;
                }
            }
        }
        out.len() - mark
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spec's measured vocab: `{[UNK]: 0, ab: 1, ##cd: 2}`.
    fn small() -> WordPiece {
        WordPiece::new(
            [
                ("[UNK]".to_string(), TokenId(0)),
                ("ab".to_string(), TokenId(1)),
                ("##cd".to_string(), TokenId(2)),
            ],
            "[UNK]",
            "##",
            100,
        )
        .unwrap()
    }

    fn enc(wp: &WordPiece, s: &str) -> Vec<u32> {
        let mut out = Vec::new();
        wp.encode_unit(s.as_bytes(), &mut out);
        out.into_iter().map(|t| t.0).collect()
    }

    /// Every case measured against `tokenizers` 0.22.2.
    #[test]
    fn wordpiece_maxmatch_measured_cases() {
        let wp = small();
        assert_eq!(enc(&wp, "abcd"), vec![1, 2]);
        assert_eq!(enc(&wp, "ab"), vec![1]);
        // Failure at any position discards the pieces already matched.
        assert_eq!(enc(&wp, "abce"), vec![0]);
        assert_eq!(enc(&wp, "abc"), vec![0]);
        assert_eq!(enc(&wp, "abcdab"), vec![0]);
        // A head piece is not usable as a continuation, and vice versa.
        assert_eq!(enc(&wp, "cd"), vec![0]);
        // Empty emits nothing at all — not an UNK.
        assert_eq!(enc(&wp, ""), Vec::<u32>::new());
    }

    /// `unaffable → un ##aff ##able`, the canonical BERT example.
    #[test]
    fn wordpiece_longest_match_first() {
        let wp = WordPiece::new(
            [
                ("[UNK]".to_string(), TokenId(0)),
                ("un".to_string(), TokenId(1)),
                ("##aff".to_string(), TokenId(2)),
                ("##able".to_string(), TokenId(3)),
                ("##a".to_string(), TokenId(4)),
            ],
            "[UNK]",
            "##",
            100,
        )
        .unwrap();
        assert_eq!(enc(&wp, "unaffable"), vec![1, 2, 3]);
    }

    /// The cap counts chars, not bytes: 4 × `é` is 8 bytes and 4 chars, so it
    /// passes at 4 and fails at 3. Measured both ways.
    #[test]
    fn wordpiece_length_cap_counts_chars() {
        let vocab = || {
            [
                ("[UNK]".to_string(), TokenId(0)),
                ("é".to_string(), TokenId(1)),
                ("##é".to_string(), TokenId(2)),
            ]
        };
        let at4 = WordPiece::new(vocab(), "[UNK]", "##", 4).unwrap();
        assert_eq!(enc(&at4, "éééé"), vec![1, 2, 2, 2]);
        let at3 = WordPiece::new(vocab(), "[UNK]", "##", 3).unwrap();
        assert_eq!(enc(&at3, "éééé"), vec![0]);

        let ascii = WordPiece::new(
            [
                ("[UNK]".to_string(), TokenId(0)),
                ("a".to_string(), TokenId(1)),
                ("##a".to_string(), TokenId(2)),
            ],
            "[UNK]",
            "##",
            4,
        )
        .unwrap();
        assert_eq!(enc(&ascii, "aaaa"), vec![1, 2, 2, 2]);
        assert_eq!(enc(&ascii, "aaaaa"), vec![0]);
    }

    /// A non-`##` prefix must work, and a piece is classified by prefix at
    /// load time rather than by lookup order.
    #[test]
    fn wordpiece_custom_prefix() {
        let wp = WordPiece::new(
            [
                ("<unk>".to_string(), TokenId(0)),
                ("foo".to_string(), TokenId(1)),
                ("@@bar".to_string(), TokenId(2)),
            ],
            "<unk>",
            "@@",
            100,
        )
        .unwrap();
        assert_eq!(enc(&wp, "foobar"), vec![1, 2]);
        assert_eq!(wp.prefix(), "@@");
    }

    /// Multi-byte candidates must not be split mid-character by the byte-wise
    /// backoff: with `é` (2 bytes) in the vocab, `"éa"` must find `é` even
    /// though the backoff probes the 1-byte prefix 0xC3 on the way down.
    #[test]
    fn wordpiece_bytewise_backoff_cannot_match_partial_char() {
        let wp = WordPiece::new(
            [
                ("[UNK]".to_string(), TokenId(0)),
                ("é".to_string(), TokenId(1)),
                ("##a".to_string(), TokenId(2)),
            ],
            "[UNK]",
            "##",
            100,
        )
        .unwrap();
        assert_eq!(enc(&wp, "éa"), vec![1, 2]);
    }

    /// A vocab without its own unk token is a load error, not a panic at
    /// encode time.
    #[test]
    fn wordpiece_missing_unk_is_an_error() {
        let err = WordPiece::new([("a".to_string(), TokenId(0))], "[UNK]", "##", 100)
            .expect_err("should reject");
        assert!(err.to_string().contains("[UNK]"), "{err}");
    }
}
