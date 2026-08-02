//! Fast pretokenizer for the SuperBPE stage-1 regex (Liu et al., 2025 —
//! the pattern `scripts/train_tokenizer.sh` passes as `--regex_string`):
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! The o200k scheme without contraction suffixes: identical to the Nemotron
//! pattern except for `\p{N}{1,3}` in place of `\p{N}`. See `o200k_family`
//! (`CONTRACTIONS = false`, `DIGITS3 = true`).
//!
//! Unlike the GPT-2 (r50k) scheme, the letter-run classes here include
//! `\p{M}`, so a combining mark stays inside the run it attaches to. That
//! matters for any script that writes vowels as marks: under r50k's
//! ` ?\p{L}+` the Devanagari word `हिन्दी` splits into six pretokens (each
//! matra and the virama falls out of `\p{L}` into the punctuation branch) and
//! BPE, which cannot merge across pretoken boundaries, can never learn a
//! consonant+matra unit. Under this scheme it is a single pretoken. See
//! `stage1_keeps_devanagari_whole`.

use super::mask::{MaskScheme, MaskState};
use super::o200k_family;
use crate::pretokenize::Pretoken;

pub(crate) struct SuperBPEStage1Scheme;

impl MaskScheme for SuperBPEStage1Scheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        o200k_family::advance_pos::<false, true, true, false>(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        o200k_family::batch_masks::<false, true, true, false>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { o200k_family::batch_masks_x86::<AVX512, false, true, true, false>(bytes, scan) }
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512/AVX2 detected at
/// runtime), iteration runs the shared o200k-family mask scanner (see
/// `o200k_family::batch_masks`); elsewhere every token takes the scalar
/// `advance_pos`.
pub struct FastSuperBPEStage1Pretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
}

impl<'a> FastSuperBPEStage1Pretokenizer<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self::with_pos(bytes, 0)
    }

    /// Resume iteration at a byte offset previously returned by [`Self::pos`].
    #[inline]
    pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, state: MaskState::new(pos) }
    }

    /// Current position as a byte offset into the input.
    #[inline]
    pub fn pos(&self) -> usize {
        self.state.pos
    }
}

impl<'a> Iterator for FastSuperBPEStage1Pretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.state.next_span::<SuperBPEStage1Scheme>(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

super::impl_mask_pretoken_spans!(FastSuperBPEStage1Pretokenizer, SuperBPEStage1Scheme);

#[cfg(test)]
mod tests {
    use super::*;

    /// The SuperBPE stage-1 pattern verbatim, as
    /// `scripts/run_official_tokenizer_benchmark.py:30-37` assembles it — no
    /// possessive quantifiers, so it runs directly under fancy-regex.
    const STAGE1_REF_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn regex_tokens(s: &str) -> Vec<String> {
        let re = fancy_regex::Regex::new(STAGE1_REF_REGEX).unwrap();
        re.find_iter(s)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    }

    fn fast_tokens(s: &str) -> Vec<String> {
        FastSuperBPEStage1Pretokenizer::new(s.as_bytes())
            .map(|t| String::from_utf8_lossy(t.0).into_owned())
            .collect()
    }

    /// The o200k small-case list applies verbatim (contraction cases just
    /// tokenize differently, which the reference regex reflects).
    #[test]
    fn stage1_small_cases() {
        for case in crate::pretokenize::fast::o200k::tests::SMALL_CASES {
            assert_eq!(
                fast_tokens(case),
                regex_tokens(case),
                "Mismatch on case {case:?}"
            );
        }
    }

    /// Random codepoint soup drawn from classes the scheme distinguishes,
    /// compared against the reference regex.
    #[test]
    fn stage1_matches_regex_random() {
        use rand::prelude::*;
        let pools: &[&[char]] = &[
            &['a', 'z', 'é', 'ß', 'ж', 'ا', '한', '日'],      // lower/caseless
            &['A', 'Z', 'É', 'Ж', 'Ǆ', 'ǅ'],                  // upper/title
            &['1', '9', '٢', '½', 'Ⅷ', '๕'],                // numbers
            &[' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}'],   // whitespace
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}'],   // marks
            &['.', ',', '!', '$', '\'', '«', '¡', '€', '☃', '/'], // punct/symbols
            &['\u{0}', '\u{ad}', '\u{200b}', '\u{e0001}'],    // other (C*)
        ];
        let mut rng = StdRng::seed_from_u64(0x5B9E_1A57);
        for round in 0..3000 {
            let len = rng.random_range(1..40);
            let s: String = (0..len)
                .map(|_| {
                    let pool = pools.choose(&mut rng).unwrap();
                    *pool.choose(&mut rng).unwrap()
                })
                .collect();
            assert_eq!(
                fast_tokens(&s),
                regex_tokens(&s),
                "Mismatch on round {round}, case {s:?}"
            );
        }
    }

    /// Devanagari combining marks stay inside their letter run here, but fall
    /// out of r50k's ` ?\p{L}+` into the punctuation branch. `हिन्दी` is
    /// ह(Lo) ि(Mc) न(Lo) ्(Mn) द(Lo) ी(Mc): one pretoken under stage 1, six
    /// under GPT-2. This is the whole reason the trainers need a scheme
    /// argument — see the `pretokenizer` parameter on `train_superbpe`.
    #[test]
    fn stage1_keeps_devanagari_whole() {
        let hindi = "हिन्दी";
        assert_eq!(fast_tokens(hindi), vec![hindi.to_string()]);
        assert_eq!(fast_tokens(hindi), regex_tokens(hindi));

        let r50k: Vec<String> = crate::pretokenize::fast::FastR50kPretokenizer::new(hindi.as_bytes())
            .map(|t| String::from_utf8_lossy(t.0).into_owned())
            .collect();
        assert_eq!(r50k, vec!["ह", "ि", "न", "्", "द", "ी"]);
    }
}
