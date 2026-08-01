//! "Superword" pretokenizer: the whitespace-lifted scheme a SuperBPE
//! tokenizer needs at inference.
//!
//! SuperBPE (Liu et al., 2025) trains subwords with ordinary whitespace
//! pretokenization, then lifts that restriction and keeps merging, learning
//! "superword" tokens that bridge whitespace. A tokenizer exported that way
//! carries a `ByteLevel { use_regex: false }` pre_tokenizer, i.e. *no*
//! splitting at all: the whole segment (a document, or the text between two
//! added-token matches) is one pretoken, and BPE — using the learned
//! superword merges — runs across it. That is exactly what HF `tokenizers`
//! does for a `use_regex=false` ByteLevel BPE, so gigatoken matches it token
//! for token (see the differential parity test).
//!
//! This scalar "yield the whole segment" pretokenizer is deliberately
//! simple: the SIMD boundary scanners the other schemes use exist to *find*
//! split points, and this scheme has none. gigatoken's other machinery —
//! parallel document splitting on the separator, the pooled workers, the
//! merge core — still applies, so SuperBPE encoding is fast without a
//! bespoke vectorized scanner. (The digit-splitting stage-2 regex some
//! released SuperBPE tokenizers bake in is a separate scheme; it is not
//! handled here.)

use crate::pretokenize::Pretoken;

/// Yields the input segment as a single pretoken (no interior splits): the
/// `ByteLevel { use_regex: false }` behavior SuperBPE tokenizers rely on.
pub struct FastSuperwordPretokenizer<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> FastSuperwordPretokenizer<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self::with_pos(bytes, 0)
    }

    /// Resume at a byte offset previously returned by [`Self::pos`] (parity
    /// with the other fast pretokenizers' bindings entry point).
    #[inline]
    pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }
}

impl<'a> Iterator for FastSuperwordPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        if self.pos >= self.bytes.len() {
            return None;
        }
        let start = self.pos;
        self.pos = self.bytes.len();
        Some(Pretoken(&self.bytes[start..]))
    }
}

// SAFETY: `fill_spans_keyed_with` writes exactly the first `n` entries from
// live spans of `self.bytes` (the single whole-segment span this yields).
unsafe impl<'a> crate::pretokenize::PretokenSpans<'a> for FastSuperwordPretokenizer<'a> {
    #[inline]
    fn fill_spans_keyed(
        &mut self,
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        crate::pretokenize::fill_spans_keyed_with(|| self.next().map(|p| p.0), batch, prefetch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yields_whole_segment_as_one_pretoken() {
        let cases: &[&[u8]] = &[
            b"",
            b"a",
            b"hello world",
            b"the quick brown fox jumps over the lazy dog",
            "multi\nline\ntext with spaces".as_bytes(),
        ];
        for &case in cases {
            let toks: Vec<&[u8]> = FastSuperwordPretokenizer::new(case).map(|p| p.0).collect();
            if case.is_empty() {
                assert!(toks.is_empty(), "empty input yields no pretokens");
            } else {
                assert_eq!(toks, vec![case], "must yield the whole segment unsplit");
            }
        }
    }

    /// The chunked span source must reproduce the iterator (one whole-segment
    /// span, keyed via the shared helper), like the other schemes.
    #[test]
    fn fill_spans_matches_iterator() {
        use crate::pretokenize::{PretokenSpans, SpanBatch, PRETOKEN_CHUNK};
        let input = b"a superword spanning many spaces here and there";
        let mut src = FastSuperwordPretokenizer::new(input);
        let mut batch = SpanBatch::new();
        let mut got: Vec<&[u8]> = Vec::new();
        loop {
            let n = src.fill_spans_keyed(&mut batch, &|_h| {});
            for i in 0..n {
                // SAFETY: i < n, the count just returned by the fill.
                got.push(unsafe { batch.span(i) });
            }
            if n < PRETOKEN_CHUNK {
                break;
            }
        }
        assert_eq!(got, vec![&input[..]]);
    }
}
