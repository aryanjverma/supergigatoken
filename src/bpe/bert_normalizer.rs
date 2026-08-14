//! HF's `BertNormalizer`, materialised.
//!
//! Applied to each added-token-delimited segment *before* pretokenization,
//! which is HF's own order (`normalize(whole text) → pre_tokenize → model`).
//! Materialising it — rather than fusing it into the pretokenizer and keying
//! the cache on raw bytes — is a deliberate choice recorded in
//! `docs/superpowers/specs/2026-08-10-fast-wordpiece-design.md`: it is exactly
//! HF's semantics with no edge cases, and `handle_chinese_chars` falls out for
//! free because the spaces it inserts are dropped by the pretokenizer's
//! whitespace split, so each CJK char becomes its own pretoken without the
//! walker knowing anything about CJK.
//!
//! The four steps run in this order, and the order is load-bearing:
//!
//! 1. `clean_text` — drop NUL, U+FFFD and control chars; map the remaining
//!    White_Space chars to U+0020. Removal wins over mapping, so U+0085 (both
//!    a control and White_Space) disappears rather than becoming a space.
//! 2. `handle_chinese_chars` — replace each CJK char `c` with `" c "`.
//! 3. `strip_accents` — NFD, then drop `Mn`. Defaults to the value of
//!    `lowercase` when the JSON says `null`, which is how `bert-base-uncased`
//!    (`strip_accents: null, lowercase: true`) ends up stripping accents.
//! 4. `lowercase` — per-char `char::to_lowercase`, which may expand.
//!
//! Every claim above and every test case below was measured against
//! `tokenizers` 0.22.2.

use crate::pretokenize::unicode::{BertNormClass, BertNormTable, is_bert_cjk};

/// Reusable scratch for [`BertNormalizer::normalize`]. Three buffers rather
/// than two: NFD reads one and writes another, and the drop-marks/lowercase
/// pass then reads *that* one, so with two buffers the middle stage would have
/// to alias its own input.
#[derive(Default)]
pub struct BertScratch {
    clean: String,
    nfd: String,
    folded: String,
    /// Output buffer for the printable-ASCII fast path, which works on bytes.
    ascii: Vec<u8>,
}

pub struct BertNormalizer {
    clean_text: bool,
    handle_chinese_chars: bool,
    /// Already resolved from `Option<bool>` at construction, so the encode path
    /// never re-derives the `null` ⇒ `lowercase` default.
    strip_accents: bool,
    lowercase: bool,
}

impl std::fmt::Debug for BertNormalizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BertNormalizer {{ clean_text: {}, handle_chinese_chars: {}, strip_accents: {}, lowercase: {} }}",
            self.clean_text, self.handle_chinese_chars, self.strip_accents, self.lowercase
        )
    }
}

impl BertNormalizer {
    /// `strip_accents: None` is HF's `null`, which resolves to `lowercase`.
    /// Measured: `lowercase=true, strip_accents=null` gives `Café → cafe`;
    /// `lowercase=false, strip_accents=null` leaves `Café` alone.
    pub fn new(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    ) -> Self {
        BertNormalizer {
            clean_text,
            handle_chinese_chars,
            strip_accents: strip_accents.unwrap_or(lowercase),
            lowercase,
        }
    }

    /// True when every step is disabled, so the loader can skip installing it.
    pub fn is_noop(&self) -> bool {
        !self.clean_text && !self.handle_chinese_chars && !self.strip_accents && !self.lowercase
    }

    /// Normalize one segment, borrowing the input when nothing changes (the
    /// shape [`crate::bpe::tiktoken`]'s `nfc_segment` uses).
    ///
    /// Invalid UTF-8 passes through untouched: HF only ever sees `&str`, so
    /// there is no parity behaviour to match, and the walkers downstream are
    /// safe on garbage by contract.
    pub fn normalize<'s>(&self, seg: &'s [u8], scratch: &'s mut BertScratch) -> &'s [u8] {
        if self.is_noop() || seg.is_empty() {
            return seg;
        }
        // Whole segment printable ASCII: every step but `lowercase` is the
        // identity, and that one is a byte map.
        if self.is_printable_ascii(seg) {
            if !self.lowercase || !seg.iter().any(u8::is_ascii_uppercase) {
                return seg;
            }
            let out = &mut scratch.ascii;
            out.clear();
            out.extend(seg.iter().map(u8::to_ascii_lowercase));
            return out;
        }
        let Ok(s) = std::str::from_utf8(seg) else {
            return seg;
        };
        let BertScratch { clean, nfd, folded, .. } = scratch;

        // Steps 1, 2, and — for ASCII only — 4, in one pass that bulk-copies
        // runs of printable ASCII instead of walking them char by char.
        //
        // This shape is load-bearing, not a micro-optimization. Measured on
        // 32 MB of OWT, single-threaded, best of 5: the char-by-char version of
        // *this pass alone* cost 47.2% of end-to-end encode (271.0 MB/s with no
        // normalizer vs 143.2 with `clean_text` only), dwarfing NFD (+0.2%) and
        // the fold (+10%). A whole-segment ASCII precheck does not help, because
        // one newline anywhere disqualifies a 1 MB document and OWT documents
        // all contain newlines — the win only arrives per *run*.
        //
        // Folding ASCII here rather than in the fold pass below is safe: NFD
        // neither decomposes nor reorders an ASCII starter, no ASCII char is
        // `Mn`, and `to_lowercase` maps ASCII to ASCII — so the three steps
        // between commute with an ASCII case fold, and folding twice is
        // idempotent anyway.
        let fold_ascii = self.lowercase;
        let mut saw_non_ascii = false;
        let cleaned: &str = if self.clean_text || self.handle_chinese_chars {
            clean.clear();
            clean.reserve(s.len());
            let table = BertNormTable::get();
            let bytes = s.as_bytes();
            let mut i = 0usize;
            while i < bytes.len() {
                // Bulk: the next run of printable ASCII passes through
                // clean_text and handle_chinese_chars untouched.
                let run = i + printable_ascii_run(&bytes[i..]);
                if run > i {
                    let chunk = &bytes[i..run];
                    if fold_ascii {
                        // SAFETY: `chunk` is printable ASCII and
                        // `to_ascii_lowercase` maps ASCII to ASCII, so every byte
                        // appended is a one-byte UTF-8 sequence and `clean`
                        // remains valid UTF-8.
                        unsafe { clean.as_mut_vec() }
                            .extend(chunk.iter().map(|b| b.to_ascii_lowercase()));
                    } else {
                        // SAFETY: `chunk` is printable ASCII, hence valid UTF-8.
                        clean.push_str(unsafe { std::str::from_utf8_unchecked(chunk) });
                    }
                    i = run;
                    continue;
                }
                // One non-printable-ASCII or non-ASCII char, per HF's rules.
                let c = s[i..].chars().next().expect("i is a char boundary");
                i += c.len_utf8();
                saw_non_ascii |= !c.is_ascii();
                if self.clean_text {
                    match table.class_of(c as u32) {
                        BertNormClass::Remove => continue,
                        BertNormClass::Whitespace => {
                            clean.push(' ');
                            continue;
                        }
                        _ => {}
                    }
                }
                if self.handle_chinese_chars && is_bert_cjk(c as u32) {
                    // Measured: adjacent CJK yields a double space
                    // (`"中文x"` → `" 中  文 x"`). The pretokenizer drops runs
                    // of whitespace, so the doubling is invisible downstream —
                    // but it is what HF produces, and `normalize_str` parity
                    // tests see it.
                    clean.push(' ');
                    clean.push(c);
                    clean.push(' ');
                } else if fold_ascii && c.is_ascii() {
                    clean.push(c.to_ascii_lowercase());
                } else {
                    clean.push(c);
                }
            }
            // All-ASCII output: NFD is the identity, nothing is `Mn`, and the
            // case fold already happened above, so both passes below are dead.
            if !saw_non_ascii {
                return clean.as_bytes();
            }
            clean.as_str()
        } else {
            s
        };

        // Step 3's decomposition. Really canonical decomposition, not just mark
        // removal: measured U+2126 OHM SIGN → U+03A9 (a singleton
        // decomposition, no marks involved). Compatibility decompositions are
        // *not* applied — U+01C5 → U+01C6 comes from `to_lowercase`, not NFD.
        let decomposed: &str = if self.strip_accents {
            nfd.clear();
            icu::normalizer::DecomposingNormalizer::new_nfd()
                .normalize_to(cleaned, nfd)
                .expect("writing to a String cannot fail");
            nfd.as_str()
        } else {
            cleaned
        };

        // Steps 3b and 4 fuse: HF filters `Mn` and *then* lowercases the
        // survivors, which is per-char in both halves, so dropping and folding
        // in one pass gives the same string.
        if !self.strip_accents && !self.lowercase {
            return decomposed.as_bytes();
        }
        folded.clear();
        folded.reserve(decomposed.len());
        let table = BertNormTable::get();
        let bytes = decomposed.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // Bulk again, on the same reasoning as the clean pass: no ASCII char
            // is `Mn`, so the filter never fires inside an ASCII run, and the
            // case fold is a byte map there.
            //
            // The fold cannot be skipped outright even when the clean pass
            // already folded ASCII: NFD *creates* ASCII that never went through
            // it — "É" decomposes to "E" + U+0301, and the "E" still needs
            // lowercasing. Folding an already-folded byte is idempotent, so
            // applying the map unconditionally is both correct and cheaper than
            // tracking which bytes are new.
            let run = i + ascii_run(&bytes[i..]);
            if run > i {
                let chunk = &bytes[i..run];
                if self.lowercase {
                    // SAFETY: ASCII in, ASCII out — `folded` stays valid UTF-8.
                    unsafe { folded.as_mut_vec() }
                        .extend(chunk.iter().map(|b| b.to_ascii_lowercase()));
                } else {
                    // SAFETY: an all-ASCII slice is valid UTF-8.
                    folded.push_str(unsafe { std::str::from_utf8_unchecked(chunk) });
                }
                i = run;
                continue;
            }
            let c = decomposed[i..].chars().next().expect("i is a char boundary");
            i += c.len_utf8();
            if self.strip_accents && table.class_of(c as u32) == BertNormClass::Mark {
                continue;
            }
            match self.lowercase {
                // `to_lowercase` may expand: measured U+0130 İ → "i" + U+0307
                // when lowercasing without stripping accents.
                true => folded.extend(c.to_lowercase()),
                false => folded.push(c),
            }
        }
        folded.as_bytes()
    }

    /// Reference implementation of the four steps, straight from HF's order with
    /// no bulk copying: one `chars()` pass per step. Kept as the differential
    /// oracle for [`Self::normalize`]'s run-based version — the repo's
    /// convention for establishing that a fast path is equivalent (see
    /// `pretokenize/reference/`) rather than arguing it.
    #[cfg(test)]
    fn normalize_reference(&self, seg: &[u8]) -> Vec<u8> {
        if self.is_noop() {
            return seg.to_vec();
        }
        let Ok(s) = std::str::from_utf8(seg) else {
            return seg.to_vec();
        };
        let table = BertNormTable::get();
        let mut out = String::new();
        if self.clean_text {
            for c in s.chars() {
                match table.class_of(c as u32) {
                    BertNormClass::Remove => {}
                    BertNormClass::Whitespace => out.push(' '),
                    _ => out.push(c),
                }
            }
        } else {
            out.push_str(s);
        }
        if self.handle_chinese_chars {
            let mut padded = String::new();
            for c in out.chars() {
                if is_bert_cjk(c as u32) {
                    padded.push(' ');
                    padded.push(c);
                    padded.push(' ');
                } else {
                    padded.push(c);
                }
            }
            out = padded;
        }
        if self.strip_accents {
            let mut nfd = String::new();
            icu::normalizer::DecomposingNormalizer::new_nfd()
                .normalize_to(&out, &mut nfd)
                .expect("writing to a String cannot fail");
            out = nfd.chars().filter(|c| table.class_of(*c as u32) != BertNormClass::Mark).collect();
        }
        if self.lowercase {
            out = out.chars().flat_map(char::to_lowercase).collect();
        }
        out.into_bytes()
    }

    /// Is every byte printable ASCII (0x20–0x7E)?
    ///
    /// For such a segment every step of the normalizer except `lowercase` is
    /// provably the identity, whatever the flags say: no byte is a control or a
    /// non-space whitespace char (so `clean_text` changes nothing), no ASCII
    /// char is CJK, ASCII is already in NFD, and no ASCII char is `Mn`. DEL
    /// (0x7F) and the control range are excluded precisely because `clean_text`
    /// *would* touch them.
    ///
    /// Uppercase deliberately does **not** disqualify a segment. An earlier
    /// version folded "contains no uppercase" into this test so the fast path
    /// could always borrow the input — but uppercase appears in nearly every
    /// real document, so the path effectively never fired and every segment paid
    /// three char-by-char passes plus an ICU NFD walk. That mistake cost 2.6× of
    /// end-to-end encode throughput; lowercasing ASCII is a byte map and belongs
    /// *inside* the fast path, not a reason to leave it.
    #[inline]
    fn is_printable_ascii(&self, seg: &[u8]) -> bool {
        use std::simd::prelude::*;
        let mut i = 0usize;
        // `wrapping_sub(0x20) > 0x5E` catches both halves (control and >= 0x7F)
        // with one unsigned compare per lane.
        while i + 16 <= seg.len() {
            let v = u8x16::from_slice(&seg[i..]);
            if (v - u8x16::splat(0x20)).simd_gt(u8x16::splat(0x5E)).any() {
                return false;
            }
            i += 16;
        }
        seg[i..].iter().all(|&b| b.wrapping_sub(0x20) <= 0x5E)
    }
}

/// Length of the leading run of ASCII bytes (< 0x80).
///
/// Wider than [`printable_ascii_run`] on purpose: the fold pass only drops marks
/// and folds case, and controls are neither, so it can treat every ASCII byte as
/// bulk — while `clean_text` has to stop on them.
#[inline]
fn ascii_run(bytes: &[u8]) -> usize {
    use std::simd::prelude::*;
    let mut i = 0usize;
    while i + 16 <= bytes.len() {
        let m = u8x16::from_slice(&bytes[i..])
            .simd_ge(u8x16::splat(0x80))
            .to_bitmask();
        if m != 0 {
            return i + m.trailing_zeros() as usize;
        }
        i += 16;
    }
    while i < bytes.len() && bytes[i] < 0x80 {
        i += 1;
    }
    i
}

/// Length of the leading run of printable-ASCII bytes (0x20–0x7E).
///
/// The same "hop to the next attention byte" scan
/// [`crate::bpe::sentencepiece::PrecompiledCharsmap::normalize_into`] uses, and
/// for the same reason: everything outside this range needs per-character
/// handling, everything inside it can be copied in bulk.
#[inline]
fn printable_ascii_run(bytes: &[u8]) -> usize {
    use std::simd::prelude::*;
    let mut i = 0usize;
    while i + 16 <= bytes.len() {
        let v = u8x16::from_slice(&bytes[i..]);
        let attention = (v - u8x16::splat(0x20)).simd_gt(u8x16::splat(0x5E));
        let m = attention.to_bitmask();
        if m != 0 {
            return i + m.trailing_zeros() as usize;
        }
        i += 16;
    }
    while i < bytes.len() && bytes[i].wrapping_sub(0x20) <= 0x5E {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(n: &BertNormalizer, s: &str) -> String {
        let mut scratch = BertScratch::default();
        String::from_utf8(n.normalize(s.as_bytes(), &mut scratch).to_vec()).unwrap()
    }

    /// `bert-base-uncased`: clean_text, handle_chinese_chars,
    /// strip_accents=null, lowercase=true.
    fn uncased() -> BertNormalizer {
        BertNormalizer::new(true, true, None, true)
    }

    /// `bert-base-cased` / multilingual-cased: the same with lowercase=false,
    /// which makes strip_accents=null resolve to false.
    fn cased() -> BertNormalizer {
        BertNormalizer::new(true, true, None, false)
    }

    /// Every row measured against `tokenizers` 0.22.2 `normalize_str`.
    #[test]
    fn bert_normalizer_measured_cases() {
        let u = uncased();
        assert_eq!(norm(&u, "Café"), "cafe");
        assert_eq!(norm(&u, "\u{2126}"), "\u{3c9}"); // OHM SIGN → greek omega
        assert_eq!(norm(&u, "\u{130}"), "i"); // İ → NFD I + U+0307 → strip → lower
        assert_eq!(norm(&u, "\u{345}"), ""); // a mark-only string normalizes to empty
        assert_eq!(norm(&u, "中文x"), " 中  文 x");
        assert_eq!(norm(&u, "unaffable"), "unaffable");
        assert_eq!(norm(&u, "ＡＢ"), "ａｂ"); // lowercase is not ASCII-only
        assert_eq!(norm(&u, "ß"), "ß"); // to_lowercase leaves it alone
        assert_eq!(norm(&u, "İstanbul"), "istanbul");

        let c = cased();
        assert_eq!(norm(&c, "Café"), "Café");
        assert_eq!(norm(&c, "\u{2126}"), "\u{2126}"); // no NFD without strip
        assert_eq!(norm(&c, "\u{130}"), "\u{130}");
        assert_eq!(norm(&c, "中文x"), " 中  文 x");

        // strip_accents explicitly on with lowercase off: NFD runs, folding
        // does not. Measured: Ω stays uppercase, İ → I.
        let strip_only = BertNormalizer::new(true, true, Some(true), false);
        assert_eq!(norm(&strip_only, "Café"), "Cafe");
        assert_eq!(norm(&strip_only, "\u{2126}"), "\u{3a9}");
        assert_eq!(norm(&strip_only, "\u{130}"), "I");
        assert_eq!(norm(&strip_only, "\u{345}"), "");

        // lowercase on, strip explicitly off: İ expands to two chars.
        let lower_only = BertNormalizer::new(true, true, Some(false), true);
        assert_eq!(norm(&lower_only, "Café"), "café");
        assert_eq!(norm(&lower_only, "\u{130}"), "i\u{307}");
        assert_eq!(norm(&lower_only, "İstanbul"), "i\u{307}stanbul");
        assert_eq!(norm(&lower_only, "\u{345}"), "\u{345}");
    }

    /// The `clean_text` table: control wins over whitespace, `Cn` is kept, and
    /// U+FFFD is removed even though it is a symbol.
    #[test]
    fn bert_normalizer_clean_text_table() {
        let n = BertNormalizer::new(true, false, Some(false), false);
        for (cp, want) in [
            (0x09u32, "a b"),
            (0x0A, "a b"),
            (0x0D, "a b"),
            (0x0B, "ab"),
            (0x0C, "ab"),
            (0x1F, "ab"),
            (0x7F, "ab"),
            (0x85, "ab"),
            (0xA0, "a b"),
            (0x2003, "a b"),
            (0x3000, "a b"),
            (0x2028, "a b"),
            (0x2029, "a b"),
            (0x200B, "ab"),
            (0x180E, "ab"),
            (0xE000, "ab"),
            (0xFFFD, "ab"),
            (0x00, "ab"),
            // The trap: U+0378 is unassigned (Cn), which HF's control
            // predicate excludes, so it survives.
            (0x0378, "a\u{378}b"),
            // Stale-UCD deltas: Cf to ICU, unknown to HF, therefore kept.
            (0x0890, "a\u{890}b"),
            (0x13430, "a\u{13430}b"),
        ] {
            let input = format!("a{}b", char::from_u32(cp).unwrap());
            assert_eq!(norm(&n, &input), want, "U+{cp:04X}");
        }
    }

    /// CJK padding covers the range list, including the measured U+2B820 hole
    /// and the exclusion of kana.
    #[test]
    fn bert_normalizer_cjk_padding() {
        let n = BertNormalizer::new(false, true, Some(false), false);
        for cp in [0x4E00u32, 0x9FFF, 0x3400, 0xF900, 0x20000, 0x2B81F, 0x2B920, 0x2F800] {
            let c = char::from_u32(cp).unwrap();
            assert_eq!(norm(&n, &format!("a{c}b")), format!("a {c} b"), "U+{cp:04X}");
        }
        for cp in [0x2B820u32, 0x3040, 0x30FF, 0x2CEB0] {
            let c = char::from_u32(cp).unwrap();
            assert_eq!(norm(&n, &format!("a{c}b")), format!("a{c}b"), "U+{cp:04X}");
        }
    }

    /// The printable-ASCII fast path must agree with the general path, and must
    /// borrow the input outright when even the lowercase step is a no-op.
    #[test]
    fn bert_normalizer_ascii_fast_path() {
        let u = uncased();
        let mut scratch = BertScratch::default();
        let seg = b"already lowercase, with punctuation!";
        let got = u.normalize(seg, &mut scratch);
        assert!(std::ptr::eq(got.as_ptr(), seg.as_ptr()), "should borrow");
        assert_eq!(got, seg);

        // Uppercase stays on the fast path (it is a byte map, not a reason to
        // leave) — the regression this test exists for.
        assert_eq!(norm(&u, "Has Uppercase"), "has uppercase");
        assert_eq!(norm(&u, "MIXED Case 123!"), "mixed case 123!");
        // Without `lowercase` the same segment is borrowed unchanged.
        let mixed = b"Has Uppercase";
        let c = cased();
        let mut s2 = BertScratch::default();
        assert!(std::ptr::eq(c.normalize(mixed, &mut s2).as_ptr(), mixed.as_ptr()));

        // A control byte leaves the fast path even though the text is ASCII.
        assert_eq!(norm(&u, "a\u{b}b"), "ab");
        // And DEL, which clean_text deletes.
        assert_eq!(norm(&u, "a\u{7f}b"), "ab");
        // Tab is ASCII but not printable: clean_text maps it to a space, so it
        // must take the general path.
        assert_eq!(norm(&u, "a\tB"), "a b");
    }

    /// A long ASCII run exercises the 16-byte SIMD stride and its scalar tail
    /// at every alignment.
    #[test]
    fn bert_normalizer_ascii_scan_all_lengths() {
        let u = uncased();
        for len in 0..80usize {
            let lower: String = std::iter::repeat_n('a', len).collect();
            assert_eq!(norm(&u, &lower), lower, "len {len}");
            // One uppercase at each position must be found by the scan.
            for pos in 0..len {
                let mut s: Vec<u8> = lower.clone().into_bytes();
                s[pos] = b'Q';
                let want = String::from_utf8(s.clone()).unwrap().to_lowercase();
                assert_eq!(norm(&u, std::str::from_utf8(&s).unwrap()), want, "len {len} pos {pos}");
            }
        }
    }

    #[test]
    fn bert_normalizer_noop_and_invalid_utf8() {
        let n = BertNormalizer::new(false, false, Some(false), false);
        assert!(n.is_noop());
        assert_eq!(norm(&n, "Anything Ünchanged"), "Anything Ünchanged");

        // Invalid UTF-8 passes through rather than panicking.
        let u = uncased();
        let mut scratch = BertScratch::default();
        let garbage: &[u8] = b"\xff\xfe\x80A";
        assert_eq!(u.normalize(garbage, &mut scratch), garbage);
    }

    /// The run-based [`BertNormalizer::normalize`] must agree with the
    /// straightforward four-pass reference on random text, for every flag
    /// combination. This is what licenses the bulk ASCII copy and the early
    /// return that skips NFD — both are arguments about where ASCII runs can be
    /// treated as opaque, and arguments are what this test replaces.
    #[test]
    fn bert_normalizer_matches_reference_random() {
        use rand::prelude::*;
        let pools: &[&[char]] = &[
            // Weighted toward ASCII, like real text, so the bulk path dominates
            // and the boundaries between runs are where the cases land.
            &['a', 'e', 'i', 'z', ' ', 'T', 'Q', '.', ',', '!', '-', '0', '9', '\'', '"'],
            &['\n', '\t', '\r', '\u{b}', '\u{c}', '\u{85}', '\u{7f}', '\u{0}', '\u{1f}'],
            &['é', 'ï', 'Ω', 'İ', 'ß', 'ﬁ', 'Ǆ', 'ǅ', 'ﬀ'],
            &['\u{301}', '\u{345}', '\u{5bf}', '\u{93b}', '\u{111c9}', '\u{7fd}', '\u{1734}'],
            &['中', '文', '\u{3400}', '\u{2b820}', '\u{f900}', 'テ', 'ス'],
            &['\u{a0}', '\u{2003}', '\u{2028}', '\u{3000}', '\u{200b}', '\u{feff}', '\u{378}'],
            &['\u{fffd}', '\u{e000}', '\u{2260}', '\u{20ac}', '\u{166d}'],
        ];
        let mut rng = StdRng::seed_from_u64(0xB3E7_0000);
        let mut scratch = BertScratch::default();
        for (lower, strip) in [
            (true, None),
            (false, None),
            (true, Some(true)),
            (true, Some(false)),
            (false, Some(true)),
            (false, Some(false)),
        ] {
            for clean in [true, false] {
                for cjk in [true, false] {
                    let n = BertNormalizer::new(clean, cjk, strip, lower);
                    for round in 0..400 {
                        // Long enough to cross the 16-byte SIMD stride several
                        // times, with runs of ASCII between the odd characters.
                        let len = rng.random_range(1..90);
                        let s: String = (0..len)
                            .map(|_| {
                                // 70% plain ASCII: produces multi-byte runs.
                                let pool = if rng.random_bool(0.7) {
                                    pools[0]
                                } else {
                                    *pools.choose(&mut rng).unwrap()
                                };
                                *pool.choose(&mut rng).unwrap()
                            })
                            .collect();
                        let got = n.normalize(s.as_bytes(), &mut scratch).to_vec();
                        let want = n.normalize_reference(s.as_bytes());
                        assert_eq!(
                            String::from_utf8_lossy(&got),
                            String::from_utf8_lossy(&want),
                            "round {round}, clean={clean} cjk={cjk} lower={lower} strip={strip:?}, input {s:?}"
                        );
                    }
                }
            }
        }
    }

    /// Scratch reuse across calls must not leak state between segments.
    #[test]
    fn bert_normalizer_scratch_reuse() {
        let u = uncased();
        let mut scratch = BertScratch::default();
        for _ in 0..3 {
            let a = u.normalize("Café".as_bytes(), &mut scratch).to_vec();
            assert_eq!(a, b"cafe");
            let b = u.normalize("中文".as_bytes(), &mut scratch).to_vec();
            assert_eq!(String::from_utf8(b).unwrap(), " 中  文 ");
        }
    }
}
