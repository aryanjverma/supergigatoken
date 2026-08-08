//! Fast scalar pretokenizer for the released SuperBPE inference scheme — a
//! `Sequence` of one `Split` (Isolated) plus `ByteLevel { use_regex: false }`:
//! `\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)`
//!
//! This is the SuperBPE stage-1 regex with the word alternatives deleted,
//! keeping only the ones that *bound* a runaway token: digit runs three at a
//! time, punctuation runs of length 2+, and trailing space runs. Letters and
//! lone punctuation are therefore never split off — `"hi! there"` is one
//! piece — which is exactly the point: a superword may span whitespace.
//!
//! The regex is not full-coverage, so HF's `Isolated` behavior emits the
//! matches **plus the gaps between them** (see [`scan_gap_from`]; `deepseek_v3`
//! faces the same shape). Alternation is leftmost-first with backtracking,
//! which shows up in exactly one place: ` +(?!\S)` before a non-whitespace
//! char gives back its last space ([`space_run_end`]).
//!
//! No alternative can start with `\p{L}`, so a gap can scan ASCII letters with
//! SWAR without checking for an alternative start.
//!
//! Scalar, not a [`super::mask::MaskScheme`]: the gap-end test is a per-char
//! class test against `\p{N}` ∪ `[^\s\p{L}\p{N}]` ∪ `' '`, and space is the
//! most common byte in prose, so there is no long-skip `memchr` win to be had.
//! Whether the per-byte cost is worth a SIMD scanner is recorded as a
//! measurement in `pretokenizer_optimization_log.md`, not guessed at here.

use super::{decode_cp, scan_numbers_max3, scan_other_from, swar_scan_letters};
use crate::pretokenize::Pretoken;
use crate::pretokenize::unicode::{CharClass, class_of};

pub struct FastSuperwordBoundedPretokenizer<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> FastSuperwordBoundedPretokenizer<'a> {
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

impl<'a> Iterator for FastSuperwordBoundedPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        if self.pos >= self.bytes.len() {
            return None;
        }
        let start = self.pos;
        self.pos = advance_pos(self.bytes, start);
        Some(Pretoken(&self.bytes[start..self.pos]))
    }
}

// SAFETY: delegates to `fill_spans_keyed_with_buf`, which writes exactly the
// first `n` entries from live in-bounds spans of `self.bytes`. `advance_pos`
// returns a strictly greater offset that never exceeds `bytes.len()`, so every
// span is nonempty and in bounds.
unsafe impl<'a> crate::pretokenize::PretokenSpans<'a> for FastSuperwordBoundedPretokenizer<'a> {
    #[inline(never)]
    fn fill_spans_keyed(
        &mut self,
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        let (bytes, len) = (self.bytes, self.bytes.len());
        let mut pos = self.pos;
        let n = crate::pretokenize::fill_spans_keyed_with_buf(
            bytes,
            || {
                if pos >= len {
                    return None;
                }
                let start = pos;
                pos = advance_pos(bytes, start);
                Some((start, pos))
            },
            batch,
            prefetch,
        );
        self.pos = pos;
        n
    }
}

/// Decode the char at `pos` as `(codepoint, byte length)`, ASCII inline.
/// `pos < bytes.len()` required; [`decode_cp`] documents the invalid-UTF-8
/// guarantees the walkers depend on (never reads or returns past `len`).
#[inline(always)]
fn cp_at(bytes: &[u8], pos: usize) -> (u32, usize) {
    // SAFETY: caller guarantees pos < bytes.len().
    let b = unsafe { *bytes.get_unchecked(pos) };
    if b < 0x80 {
        (b as u32, 1)
    } else {
        // SAFETY: pos < len and bytes[pos] >= 0x80, which is decode_cp's
        // documented precondition; it tolerates invalid sequences.
        unsafe { decode_cp(bytes, pos) }
    }
}

/// `[\r\n/]*`: the tail of alternative 2. Unlike the cl100k-family
/// [`super::scan_newlines`] this also absorbs `/`, so it needs its own scan.
/// All three bytes are ASCII and so cannot appear as a UTF-8 continuation
/// byte, which is why a byte-wise loop cannot stop mid-character.
#[inline(always)]
fn scan_nl_slash(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        // SAFETY: pos < bytes.len() checked by the loop condition.
        match unsafe { *bytes.get_unchecked(pos) } {
            b'\r' | b'\n' | b'/' => pos += 1,
            _ => break,
        }
    }
    pos
}

/// If the char at `pos` is `[^\s\p{L}\p{N}]`, the offset just past it.
/// `CharClass::Other` is exactly that complement — and it **includes
/// `\p{M}`**, which is why two combining marks form their own piece while one
/// stays inside the surrounding gap.
#[inline(always)]
fn other_end_at(bytes: &[u8], pos: usize) -> Option<usize> {
    if pos >= bytes.len() {
        return None;
    }
    let (cp, l) = cp_at(bytes, pos);
    (class_of(cp) == CharClass::Other).then_some(pos + l)
}

/// `[^\s\p{L}\p{N}]{2,}[\r\n/]*` starting at `run`. `None` when fewer than two
/// `Other` chars are present, i.e. when `{2,}` is unsatisfied — a *single*
/// punctuation char matches nothing and stays inside the surrounding gap,
/// which is why `"hi! there"` is one piece.
#[inline(always)]
fn punct_run_end(bytes: &[u8], run: usize) -> Option<usize> {
    let one = other_end_at(bytes, run)?;
    let two = scan_other_from(bytes, one);
    if two == one {
        return None;
    }
    Some(scan_nl_slash(bytes, two))
}

/// ` +(?!\S)` starting at `pos`, which the caller has established is a space.
/// Literal spaces only — `\s` in the lookahead is Unicode whitespace, so the
/// run may legitimately end at a tab or NBSP and still match.
///
/// Leftmost-first backtracking lives here: greedy ` +` overshoots when a
/// non-whitespace char follows, and the engine gives back one space at a time.
/// Only the first give-back can succeed — the char before a non-whitespace
/// char is then a space, and a space is not `\S` — so this is one subtraction
/// rather than a loop. A lone space before content therefore matches nothing
/// (`end - 1 == pos` is an empty match) and belongs to the gap.
#[inline(always)]
fn space_run_end(bytes: &[u8], pos: usize) -> Option<usize> {
    let len = bytes.len();
    let mut end = pos;
    // SAFETY: end < len checked by the loop condition.
    while end < len && unsafe { *bytes.get_unchecked(end) } == b' ' {
        end += 1;
    }
    if end == len {
        return Some(end); // `(?!\S)` succeeds at end of input
    }
    let (cp, _) = cp_at(bytes, end);
    if class_of(cp) == CharClass::Whitespace {
        return Some(end); // non-space whitespace is not `\S`
    }
    (end - 1 > pos).then_some(end - 1)
}

/// The end of the alternative matching at `pos`, or `None` when none does (so
/// `pos` starts a gap). `cp`/`l` are the char at `pos`, already decoded.
///
/// Priority is the regex's own: numbers, then the punctuation run (with its
/// optional single-space prefix), then the space run. At a space the
/// space-prefixed punctuation run and ` +(?!\S)` are mutually exclusive — the
/// former needs two `Other` chars after the space, the latter needs the next
/// char to be a space — so `or_else` is the complete ordering, not an
/// approximation of backtracking.
#[inline(always)]
fn match_cp(bytes: &[u8], pos: usize, cp: u32, l: usize) -> Option<usize> {
    match class_of(cp) {
        CharClass::Number => Some(scan_numbers_max3(bytes, pos + l, 1)),
        CharClass::Other => punct_run_end(bytes, pos),
        CharClass::Whitespace if cp == ' ' as u32 => {
            punct_run_end(bytes, pos + 1).or_else(|| space_run_end(bytes, pos))
        }
        _ => None,
    }
}

/// Unmatched-gap piece: the chars no alternative matches, emitted as one piece
/// the way HF's Isolated split leaves them. `first_len` is the byte length of
/// the char at `pos`, which the caller established starts a gap.
///
/// No alternative starts with `\p{L}`, so ASCII letters — the bulk of prose —
/// need no per-char alternative probe and go through SWAR.
#[inline(always)]
fn scan_gap_from(bytes: &[u8], pos: usize, first_len: usize) -> usize {
    let len = bytes.len();
    let mut p = pos + first_len;
    loop {
        p = swar_scan_letters(bytes, p);
        if p >= len {
            return len;
        }
        let (cp, l) = cp_at(bytes, p);
        if match_cp(bytes, p, cp, l).is_some() {
            return p; // an alternative starts here; the gap ends before it
        }
        p += l;
    }
}

/// Advance past one piece starting at `pos`. Returns the new position.
/// `pos` must be < `bytes.len()`.
#[inline(always)]
fn advance_pos(bytes: &[u8], pos: usize) -> usize {
    let (cp, l) = cp_at(bytes, pos);
    match match_cp(bytes, pos, cp, l) {
        Some(end) => end,
        None => scan_gap_from(bytes, pos, l),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::fast::deepseek_v3::tests::split_isolated;

    /// The released 128k's `Split` regex verbatim. `\r`/`\n` are regex escapes
    /// here, not literal control bytes (the JSON carried `\\r`), so this is a
    /// plain raw string — unlike `DEEPSEEK_V3_SPLIT_REGEXES[2]`, which embeds
    /// literal CR/LF.
    const SUPERWORD_BOUNDED_REF_REGEX: &str = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)";

    /// Pieces the reference regex produces under HF `Isolated` behavior.
    fn regex_tokens(s: &str) -> Vec<String> {
        let re = fancy_regex::Regex::new(SUPERWORD_BOUNDED_REF_REGEX).unwrap();
        split_isolated(&re, s)
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    fn fast_tokens(s: &str) -> Vec<String> {
        FastSuperwordBoundedPretokenizer::new(s.as_bytes())
            .map(|t| String::from_utf8_lossy(t.0).into_owned())
            .collect()
    }

    /// Measured against `tokenizers` 0.22.2 rather than reasoned about — the
    /// table in `docs/superpowers/specs/2026-08-07-superbpe-128k-support-design.md`.
    /// Asserted against *both* the walker and the fancy-regex oracle, so the
    /// oracle the random test relies on is itself pinned.
    const GROUND_TRUTH: &[(&str, &[&str])] = &[
        ("hi! there", &["hi! there"]),
        ("The quick brown fox", &["The quick brown fox"]),
        ("camelCase McDonald", &["camelCase McDonald"]),
        ("a   b", &["a", "  ", " b"]),
        ("  !!", &[" ", " !!"]),
        ("ab!!cd", &["ab", "!!", "cd"]),
        ("12345", &["123", "45"]),
        ("१२३४", &["१२३", "४"]),
        ("x!'s", &["x", "!'", "s"]),
        ("a\u{301}\u{301}b", &["a", "\u{301}\u{301}", "b"]),
        ("é\u{301}", &["é\u{301}"]),
        ("a \t b", &["a", " ", "\t b"]),
        ("!!//x", &["!!//", "x"]),
        (" ...\n\n/x", &[" ...\n\n/", "x"]),
        ("end.  ", &["end.", "  "]),
    ];

    #[test]
    fn superword_bounded_matches_ground_truth() {
        for (input, want) in GROUND_TRUTH {
            assert_eq!(regex_tokens(input), *want, "oracle disagrees on {input:?}");
            assert_eq!(fast_tokens(input), *want, "walker disagrees on {input:?}");
        }
    }

    /// Random codepoint soup drawn from the classes the regex distinguishes,
    /// including the `\p{M}` / non-ASCII-digit / NBSP / lone-`/` traps.
    #[test]
    fn superword_bounded_matches_regex_random() {
        use rand::prelude::*;
        let pools: &[&[char]] = &[
            &['a', 'z', 'é', 'ß', 'ж', 'ا', '한', '日'], // letters
            &['A', 'Z', 'É', 'Ж', 'Ǆ', 'ǅ'],             // upper/title
            &['1', '9', '٢', '½', 'Ⅷ', '๕', '１'],      // \p{N}
            &[' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}', '\u{2003}'], // \s
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}'], // \p{M}
            &['.', ',', '!', '$', '\'', '«', '¡', '€', '☃', '/', '-', '+'], // punct/symbols
            &['\u{0}', '\u{7}', '\u{ad}', '\u{200b}', '\u{feff}', '\u{e0001}'], // other (C*)
        ];
        let mut rng = StdRng::seed_from_u64(0x5C0F_BD11);
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
}
