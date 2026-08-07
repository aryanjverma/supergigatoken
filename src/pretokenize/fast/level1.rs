//! Level-1 unit splitting for SuperBPE two-level encoding, on the SIMD
//! two-phase fill.
//!
//! A level-1 unit is a stage-1 pretoken extended over every following
//! pretoken that *glues* onto it. [`glues`] is the whole rule set;
//! `bpe::superword` derives it and documents what each clause costs when it
//! is missing, since a hazard the rules do not cover lowers the derived
//! threshold rather than changing anyone's tokens.
//!
//! Two walkers produce those units, and they must agree byte for byte:
//!
//! - [`Level1Walk`] — one unit at a time. The non-SIMD path, and the
//!   fallback the chunked path drops to when a chunk yields no unit
//!   boundary at all.
//! - [`Level1Fill`] — the chunked pull, and the reason this module exists.
//!   It runs the mask scanners' [`MaskState::fill_spans_two_phase`] with
//!   `GLUE = true`, which harvests a chunk of *stage-1 pretoken* boundaries
//!   with SIMD and then hands them to [`glue_filter`] before the emission
//!   loop. The alternative — `bpe::superword::Level1Units`, an
//!   `Iterator` over the runtime `FastPretokenizerDispatch` enum — pays an
//!   enum dispatch, a non-inlined `Iterator::next` and a `next_span` refill
//!   ladder *per stage-1 pretoken*, which is the per-span overhead the
//!   subword path deleted when it moved to the two-phase fill.
//!
//! # Why the filter can run on the flat boundary buffer
//!
//! Gluing is decided at a junction from the stage-1 pretoken immediately
//! left of it and the first character of the one immediately right — never
//! from the accumulated unit (`Level1Units` keeps a separate `prev_start`
//! for exactly this reason: a rule that reads how the left side *starts*
//! would otherwise re-satisfy itself after every glue and swallow the rest
//! of the document). Phase A leaves precisely those pretoken boundaries in
//! a flat `u16` buffer, so the left operand of boundary `i` is
//! `buf[i - 1]..buf[i]` and the right one starts at `buf[i]`. Dropping the
//! glued entries in place then leaves consecutive kept boundaries that *are*
//! the units, and phase B emits them unchanged.
//!
//! Two boundaries are special:
//!
//! - a boundary at end of input is never glued, matching `Level1Units`
//!   (whose unit ends when the inner pretokenizer runs out);
//! - the last one in the buffer has no `buf[i + 1]` to bound its right
//!   operand, so unless it is that end-of-input boundary it is **deferred**
//!   — dropped here and re-derived by the next fill, which harvests from
//!   the last unit end and sees it with a successor.
//!
//! Deferring rather than re-deriving the right operand with the scheme's
//! scalar `advance` is deliberate, and it is the one place this could
//! quietly go wrong. A pure-`advance` walk is *not* the same partition as
//! the mask scanner's on invalid UTF-8 — measured at 750 of 2000 random
//! byte soups, and 0 of 2000 when the input is valid UTF-8, which is why
//! nothing in the crate had noticed. Level-1 boundaries feed a merge table,
//! so the two walkers here must produce the same garbage as `Level1Units`
//! (itself scanner-driven) on garbage input, not merely the same answer on
//! well-formed input. Neither walker calls `advance` directly for that
//! reason; both go through [`MaskState::next_span`].
//!
//! A harvest can also lose *every* boundary this way — to the deferral when
//! it found only one, or to the glue rules when a chunk glues throughout (a
//! 100 KB digit run rejoins at every one of its `\p{N}{1,3}` splits). The
//! fill then emits a single unit through [`Level1Walk`], which is the same
//! escape it uses for a pretoken too long to have any boundary inside the
//! `u16` window.
//!
//! It cannot instead return what it has and let the next fill retry:
//! [`crate::pretokenize::PretokenSpans`] defines a short fill as *input
//! exhausted*, and `Tokenizer::memoized_encode_flat` stops on one. Every
//! fill here therefore runs its refill loop until the chunk is full or the
//! input ends, and every refill iteration must advance `pending`.

use super::mask::{MaskScheme, MaskState};
use super::{is_ascii_ws, is_digit};
use crate::pretokenize::{Pretoken, PretokenizerType, SpanBatch};
use std::marker::PhantomData;

// -----------------------------------------------------------------------
// The glue rules
// -----------------------------------------------------------------------

/// Whether `next` glues onto the stage-1 pretoken `prev` that precedes it —
/// the four junction shapes a sub-threshold merge can span.
/// `bpe::superword` documents each one and the merge IDs it otherwise costs.
#[inline(always)]
pub(crate) fn glues(prev: &[u8], next: &[u8]) -> bool {
    (ends_with_whitespace(prev) && starts_with_whitespace(next))
        || prev.last() == Some(&b'\'')
        || prev.first() == Some(&b'\'')
        || next.first().is_some_and(|&b| is_digit(b))
}

/// Decode the last character of `s` and report whether it is whitespace.
///
/// `s` is a pretoken, so it is non-empty, but it need not be valid UTF-8
/// (the pretokenizers pass malformed bytes through). An undecodable tail
/// counts as whitespace, which only makes the walkers glue more — the safe
/// direction.
///
/// `char::is_whitespace` agrees with [`is_ascii_ws`] on every byte below
/// 0x80, so an ASCII tail — nearly every pretoken — skips the decode
/// entirely. `boundary_whitespace_matches_decode` pins the equivalence.
#[inline(always)]
pub(crate) fn ends_with_whitespace(s: &[u8]) -> bool {
    match s.last() {
        Some(&b) if b < 0x80 => return is_ascii_ws(b),
        None => return true,
        Some(_) => {}
    }
    ends_with_whitespace_decode(s)
}

/// [`ends_with_whitespace`]'s multi-byte tail, out of line: it runs on
/// non-ASCII pretoken ends only, and inlining its four `from_utf8` attempts
/// into the filter loop costs more than the call.
#[inline(never)]
fn ends_with_whitespace_decode(s: &[u8]) -> bool {
    for k in 1..=4.min(s.len()) {
        if let Ok(tail) = std::str::from_utf8(&s[s.len() - k..]) {
            return tail.chars().next_back().is_some_and(char::is_whitespace);
        }
    }
    true
}

/// [`ends_with_whitespace`] for the first character of `s`.
#[inline(always)]
pub(crate) fn starts_with_whitespace(s: &[u8]) -> bool {
    match s.first() {
        Some(&b) if b < 0x80 => return is_ascii_ws(b),
        None => return true,
        Some(_) => {}
    }
    starts_with_whitespace_decode(s)
}

/// [`starts_with_whitespace`]'s multi-byte head; see
/// [`ends_with_whitespace_decode`].
#[inline(never)]
fn starts_with_whitespace_decode(s: &[u8]) -> bool {
    for k in 1..=4.min(s.len()) {
        if let Ok(head) = std::str::from_utf8(&s[..k]) {
            return head.chars().next().is_some_and(char::is_whitespace);
        }
    }
    true
}

// -----------------------------------------------------------------------
// The scalar walker
// -----------------------------------------------------------------------

/// One level-1 unit at a time: the scheme's next pretoken, extended over
/// every following pretoken that [`glues`] onto it.
///
/// The monomorphic twin of `bpe::superword::Level1Units`, which walks the
/// same rules over the runtime pretokenizer enum
/// (`level1_walkers_agree_*` pins that they agree). Both exist because the
/// enum form is what `derive_threshold` probes — its scheme is a runtime
/// value — while this one is what the fill can inline.
///
/// Pretokens come from [`MaskState::next_span`], not from the scheme's
/// `advance`: the two are different partitions on invalid UTF-8, and this
/// walker has to match `Level1Units` there too (see the module docs).
pub(crate) struct Level1Walk<S: MaskScheme> {
    state: MaskState,
    /// A pretoken pulled from the scanner that did not glue onto the unit
    /// just returned, and so starts the next one.
    pending: Option<(usize, usize)>,
    scheme: PhantomData<fn() -> S>,
}

impl<S: MaskScheme> Level1Walk<S> {
    #[inline]
    pub(crate) fn new(pos: usize) -> Self {
        Level1Walk {
            state: MaskState::new(pos),
            pending: None,
            scheme: PhantomData,
        }
    }

    /// Byte offsets of the next unit, or `None` at end of input.
    #[inline]
    pub(crate) fn next_unit(&mut self, bytes: &[u8]) -> Option<(usize, usize)> {
        let (start, mut end) = match self.pending.take() {
            Some(span) => span,
            None => self.state.next_span::<S>(bytes)?,
        };
        // As in `Level1Units`: the left operand of a junction is the
        // pretoken immediately left of it, not the accumulated unit — a
        // rule that reads how the left side *starts* would otherwise
        // re-satisfy itself after every glue and swallow the document.
        let mut prev_start = start;
        while let Some((next_start, next_end)) = self.state.next_span::<S>(bytes) {
            debug_assert_eq!(next_start, end, "pretokens are consecutive");
            if !glues(&bytes[prev_start..end], &bytes[next_start..next_end]) {
                self.pending = Some((next_start, next_end));
                break;
            }
            prev_start = next_start;
            end = next_end;
        }
        Some((start, end))
    }
}

// -----------------------------------------------------------------------
// The chunked filter
// -----------------------------------------------------------------------

/// Drop the glued boundaries from a phase-A harvest, in place: `buf[0..nb]`
/// holds stage-1 pretoken ends relative to `fill_base`, ascending; on
/// return `buf[0..w]` holds the level-1 unit ends among them.
///
/// Every entry needs the pretoken on each side of it, so the last one is
/// deferred unless it ends the input — see the module docs for why it is
/// dropped rather than re-derived. `w == 0` is therefore an ordinary
/// outcome (a one-boundary harvest, or a chunk that glues throughout) and
/// the caller must handle it.
///
/// Compaction is unconditional-store plus a counter increment, so the loop
/// carries no data-dependent branch of its own (the remaining ones are the
/// non-ASCII lanes inside [`glues`], which [`ends_with_whitespace`] and
/// [`starts_with_whitespace`] keep out of line). The write cursor `w` never
/// passes the read cursor `i`, so the rewrite is safe in place.
///
/// # Safety
///
/// `buf[0..nb]` must be initialised, strictly ascending, and every entry
/// must be a pretoken end satisfying `fill_base + buf[i] <= bytes.len()`.
#[inline(always)]
pub(crate) unsafe fn glue_filter(
    bytes: &[u8],
    fill_base: usize,
    buf: *mut u16,
    nb: usize,
) -> usize {
    if nb == 0 {
        return 0;
    }
    let len = bytes.len();
    let mut w = 0usize;
    let mut prev = 0usize;
    // Entries with a known successor: their right operand is buf[i + 1].
    for i in 0..nb - 1 {
        // SAFETY: i + 1 < nb, and buf[0..nb] is initialised (fn contract).
        let end = unsafe { *buf.add(i) } as usize;
        let next_end = unsafe { *buf.add(i + 1) } as usize;
        // SAFETY: boundaries ascend and stay within `bytes` (fn contract),
        // so prev <= end <= next_end and fill_base + next_end <= len.
        let left = unsafe { bytes.get_unchecked(fill_base + prev..fill_base + end) };
        let right = unsafe { bytes.get_unchecked(fill_base + end..fill_base + next_end) };
        let keep = !glues(left, right);
        // SAFETY: w <= i < nb, inside the initialised prefix.
        unsafe { buf.add(w).write(end as u16) };
        w += keep as usize;
        prev = end;
    }
    // The final entry: kept only when it ends the input, where there is no
    // right operand and `Level1Units` ends the unit too. Otherwise deferred.
    // SAFETY: nb >= 1, so nb - 1 is in the initialised prefix.
    let last = unsafe { *buf.add(nb - 1) } as usize;
    if fill_base + last >= len {
        // SAFETY: w <= nb - 1, inside the initialised prefix.
        unsafe { buf.add(w).write(last as u16) };
        w += 1;
    }
    w
}

// -----------------------------------------------------------------------
// The span source
// -----------------------------------------------------------------------

/// Level-1 units of one segment, pulled a chunk at a time through the mask
/// scanners' two-phase fill (see the module docs). `S` is the stage-1
/// scheme; [`Level1Spans`] picks the instantiation from the plan's runtime
/// scheme once per segment.
pub(crate) struct Level1Fill<'a, S: MaskScheme> {
    bytes: &'a [u8],
    state: MaskState,
    /// The one-unit-at-a-time walker, for the arch/CPU without a SIMD
    /// scanner. Built lazily so the chunked path allocates no extra state.
    walk: Option<Level1Walk<S>>,
    scheme: PhantomData<fn() -> S>,
}

impl<'a, S: MaskScheme> Level1Fill<'a, S> {
    #[inline]
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Level1Fill {
            bytes,
            state: MaskState::new(0),
            walk: None,
            scheme: PhantomData,
        }
    }
}

/// The one-unit-at-a-time walk, as an iterator — the differential oracle
/// for the chunked fill above (`level1_walkers_agree_*`) and nothing else:
/// the encode path always pulls chunks.
impl<'a, S: MaskScheme> Iterator for Level1Fill<'a, S> {
    type Item = Pretoken<'a>;

    fn next(&mut self) -> Option<Pretoken<'a>> {
        let walk = self.walk.get_or_insert_with(|| Level1Walk::new(0));
        let (start, end) = walk.next_unit(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

// SAFETY: delegates to `fill_spans_two_phase` (which writes exactly the
// first `n` entries from spans of `self.bytes`) or to
// `fill_spans_keyed_with_buf`, whose `next` contract this upholds:
// `Level1Walk::next_unit` returns in-bounds offsets of a nonempty span.
unsafe impl<'a, S: MaskScheme> crate::pretokenize::PretokenSpans<'a> for Level1Fill<'a, S> {
    // Out of line for the same reason the mask pretokenizers' impls are:
    // the fill loop keeps its own register allocation instead of being
    // inlined into the caller's probe loop.
    #[inline(never)]
    fn fill_spans_keyed(&mut self, batch: &mut SpanBatch<'a>, prefetch: &impl Fn(u64)) -> usize {
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        if super::mask::simd_scanner_available() {
            return self
                .state
                .fill_spans_two_phase::<S, true>(self.bytes, batch, prefetch);
        }
        // No SIMD scanner: one unit at a time. The two paths are never
        // mixed on a given machine, so each owns its own cursor.
        let bytes = self.bytes;
        let walk = self.walk.get_or_insert_with(|| Level1Walk::new(0));
        crate::pretokenize::fill_spans_keyed_with_buf(
            bytes,
            || walk.next_unit(bytes),
            batch,
            prefetch,
        )
    }
}

/// [`Level1Fill`] with the stage-1 scheme chosen at run time. The match is
/// per *fill* (up to `PRETOKEN_CHUNK` units), not per unit — which is the
/// difference from routing the walk through `FastPretokenizerDispatch`,
/// whose enum arm is re-selected for every stage-1 pretoken.
///
/// `bpe::superword::CANDIDATE_SCHEMES` only ever yields the two arms below;
/// a plan built with any other scheme falls back to the iterator walk in
/// `superword_encode_segment` rather than being represented here.
pub(crate) enum Level1Spans<'a> {
    SuperBPEStage1(Level1Fill<'a, super::superbpe_stage1::SuperBPEStage1Scheme>),
    Gpt2(Level1Fill<'a, super::r50k::R50kScheme>),
}

impl<'a> Level1Spans<'a> {
    /// `None` when `scheme` has no two-phase instantiation here.
    #[inline]
    pub(crate) fn new(bytes: &'a [u8], scheme: PretokenizerType) -> Option<Self> {
        match scheme {
            PretokenizerType::SuperBPEStage1 => {
                Some(Level1Spans::SuperBPEStage1(Level1Fill::new(bytes)))
            }
            PretokenizerType::GPT2 => Some(Level1Spans::Gpt2(Level1Fill::new(bytes))),
            _ => None,
        }
    }
}

// SAFETY: both arms are `Level1Fill`, whose impl upholds the contract.
unsafe impl<'a> crate::pretokenize::PretokenSpans<'a> for Level1Spans<'a> {
    #[inline]
    fn fill_spans_keyed(&mut self, batch: &mut SpanBatch<'a>, prefetch: &impl Fn(u64)) -> usize {
        match self {
            Level1Spans::SuperBPEStage1(f) => f.fill_spans_keyed(batch, prefetch),
            Level1Spans::Gpt2(f) => f.fill_spans_keyed(batch, prefetch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::superword::Level1Units;
    use crate::pretokenize::{PRETOKEN_CHUNK, PretokenSpans};

    /// Both schemes `SuperwordPlan::build` can pick.
    const SCHEMES: [PretokenizerType; 2] =
        [PretokenizerType::SuperBPEStage1, PretokenizerType::GPT2];

    /// xorshift64: deterministic, dependency-free RNG for test inputs.
    struct XorShift64(u64);

    impl XorShift64 {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Unit lengths from the chunked fill — the path under test.
    fn fill_lens(bytes: &[u8], scheme: PretokenizerType) -> Vec<usize> {
        fill_lens_counts(bytes, scheme).0
    }

    /// [`fill_lens`] plus the per-fill span counts, so a divergence can be
    /// placed relative to the chunk boundaries.
    ///
    /// Stops on a short fill exactly as `Tokenizer::memoized_encode_flat`
    /// does, and asserts the [`crate::pretokenize::PretokenSpans`] contract
    /// that only the final fill may be short. Draining to `n == 0` instead
    /// would be a more permissive consumer than any real one — and it is:
    /// it passed a version of this fill that returned short mid-input,
    /// which silently truncated every encode.
    fn fill_lens_counts(bytes: &[u8], scheme: PretokenizerType) -> (Vec<usize>, Vec<usize>) {
        let mut spans = Level1Spans::new(bytes, scheme).expect("candidate scheme");
        let mut batch = SpanBatch::new();
        let mut lens = Vec::new();
        let mut counts = Vec::new();
        loop {
            let n = spans.fill_spans_keyed(&mut batch, &|_| {});
            for i in 0..n {
                lens.push(batch.entries[i].span_len());
            }
            if n == 0 {
                return (lens, counts);
            }
            counts.push(n);
            if n < PRETOKEN_CHUNK {
                let covered: usize = lens.iter().sum();
                assert_eq!(
                    covered,
                    bytes.len(),
                    "short fill ({n} < {PRETOKEN_CHUNK}) with {} of {} bytes consumed: \
                     PretokenSpans reads that as end of input",
                    covered,
                    bytes.len(),
                );
                return (lens, counts);
            }
        }
    }

    /// Unit lengths from `Level1Units`, the runtime-enum reference walker.
    fn reference_lens(bytes: &[u8], scheme: PretokenizerType) -> Vec<usize> {
        Level1Units::new(bytes, scheme).map(|u| u.0.len()).collect()
    }

    /// Unit lengths from [`Level1Walk`], the monomorphic one-at-a-time walk.
    fn walk_lens(bytes: &[u8], scheme: PretokenizerType) -> Vec<usize> {
        fn collect<S: MaskScheme>(bytes: &[u8]) -> Vec<usize> {
            let mut walk = Level1Walk::<S>::new(0);
            let mut lens = Vec::new();
            let mut pos = 0usize;
            while let Some((start, end)) = walk.next_unit(bytes) {
                assert_eq!(start, pos, "units must be contiguous");
                assert!(end > start && end <= bytes.len(), "unit must advance in bounds");
                lens.push(end - start);
                pos = end;
            }
            assert_eq!(pos, bytes.len(), "units must cover the input");
            lens
        }
        match scheme {
            PretokenizerType::SuperBPEStage1 => {
                collect::<super::super::superbpe_stage1::SuperBPEStage1Scheme>(bytes)
            }
            PretokenizerType::GPT2 => collect::<super::super::r50k::R50kScheme>(bytes),
            _ => unreachable!(),
        }
    }

    /// Inputs that exercise the fill's structure, not just prose: the u16
    /// boundary window (>65 KB), the `PRETOKEN_CHUNK` refill, the 64-byte
    /// batch grid, and each glue rule at scale — a digit run glues at
    /// *every* one of its `\p{N}{1,3}` splits, so a long one leaves a chunk
    /// with no unit boundary at all and must route through the scalar
    /// fallback.
    fn structural_cases() -> Vec<Vec<u8>> {
        let mut cases: Vec<Vec<u8>> = Vec::new();
        for case in [
            "",
            " ",
            "0",
            "'",
            "'s",
            "x!'s t",
            "a  b   c",
            "hello world 123 456",
            "don't 'st ''x 1'2",
            "हिन्दी текст 好 🙂\u{0301}",
            // Ordinary prose: what `superword_variants_agree` diverged on
            // while every hazard-shaped case above passed.
            "In the United States of the world, one of the most important \
             things is that the government of the people should be able to do it.",
            "def f():\n    return  1\n\n    # comment  here\n",
            "word  ,  word ;; word -- word",
            "1  2   34    567  8901",
        ] {
            cases.push(case.as_bytes().to_vec());
        }
        // Everything glues: no unit boundary in any chunk.
        cases.push("0".repeat(70_000).into_bytes());
        cases.push(" ".repeat(70_000).into_bytes());
        // One >65 KB pretoken, then ordinary text.
        cases.push(format!("{} tail", "a".repeat(70_000)).into_bytes());
        // Enough units to force several refills, with glue hazards in each.
        cases.push("don't 0 1 2  x'y ".repeat(2 * PRETOKEN_CHUNK).into_bytes());
        // The same span count at prose glue density, which is far lower:
        // how many units a harvest yields decides how many refill passes a
        // fill takes, and the hazard-dense case above exercises only one end
        // of that range.
        cases.push(
            "The quick brown fox jumps over the lazy dog, and it doesn't mind. "
                .repeat(PRETOKEN_CHUNK)
                .into_bytes(),
        );
        // Glue hazards straddling the 64-byte batch grid: an apostrophe or a
        // digit at every offset within a batch.
        for pad in 0..70 {
            cases.push(format!("{}'s 0 x", "a".repeat(pad)).into_bytes());
            cases.push(format!("{}  x", "b".repeat(pad)).into_bytes());
        }
        cases
    }

    /// The three walkers must agree unit for unit: the runtime-enum
    /// reference (`Level1Units`), the monomorphic one-at-a-time walk
    /// ([`Level1Walk`], the non-SIMD path and the fill's fallback), and the
    /// chunked SIMD fill ([`Level1Fill`]).
    ///
    /// Span lengths rather than tokens, so a divergence points at the walker
    /// that produced it instead of at an encode two levels downstream.
    #[test]
    fn level1_walkers_agree_on_structural_cases() {
        for case in structural_cases() {
            for scheme in SCHEMES {
                let want = reference_lens(&case, scheme);
                assert_eq!(
                    walk_lens(&case, scheme),
                    want,
                    "{scheme:?}: one-at-a-time walk diverged on {:?} (len {})",
                    String::from_utf8_lossy(&case[..case.len().min(80)]),
                    case.len(),
                );
                // Reported by unit index, byte offset and per-fill counts,
                // because the failure modes here are positional: whether
                // the first divergence lands on a `PRETOKEN_CHUNK` fill
                // boundary or inside one separates a chunk-carry bug from a
                // glue-rule bug, and it is invisible in a diff of two
                // thousand-element length vectors.
                let (got, counts) = fill_lens_counts(&case, scheme);
                if got != want {
                    let k = (0..got.len().min(want.len()))
                        .find(|&k| got[k] != want[k])
                        .unwrap_or(got.len().min(want.len()));
                    let off: usize = want[..k].iter().sum();
                    panic!(
                        "{scheme:?}: two-phase fill diverged on {:?} (len {})\n  \
                         first differing unit #{k} at byte {off}: got {:?}, want {:?}\n  \
                         context {:?}\n  got  {:?}\n  want {:?}\n  per-fill counts {:?}",
                        String::from_utf8_lossy(&case[..case.len().min(60)]),
                        case.len(),
                        got.get(k),
                        want.get(k),
                        String::from_utf8_lossy(
                            &case[off.saturating_sub(20)..(off + 20).min(case.len())]
                        ),
                        &got[k.saturating_sub(4)..(k + 4).min(got.len())],
                        &want[k.saturating_sub(4)..(k + 4).min(want.len())],
                        &counts[..counts.len().min(8)],
                    );
                }
            }
        }
    }

    /// The same three-way agreement on randomized input, including invalid
    /// UTF-8 and truncated multi-byte tails. Both the glue predicate's
    /// decode lanes and the harvest's bad zones key off those, and it is
    /// exactly where a walker that reached for the scheme's scalar
    /// `advance` would drift away from the scanner (module docs).
    ///
    /// Spans are placed at the END of an exactly-sized allocation, so a
    /// walker reading past the input is an observable out-of-bounds rather
    /// than a silent pass.
    #[test]
    fn level1_walkers_agree_on_random_soup() {
        const CHARS: &[&str] = &[
            "é", "ü", "好", "日", "🙂", "ß", "—", "\u{0301}", "٣", "क", "\u{2028}", "\u{00a0}",
        ];
        let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
        let iters = if cfg!(debug_assertions) { 1_500 } else { 8_000 };
        for _ in 0..iters {
            let len = (rng.next_u64() % 400) as usize;
            let mut buf: Vec<u8> = Vec::with_capacity(len);
            while buf.len() < len {
                match rng.next_u64() % 10 {
                    0..=3 => buf.push(b"abcXY z\n\t.,!?"[(rng.next_u64() % 13) as usize]),
                    4..=5 => buf.push(b"0123456789"[(rng.next_u64() % 10) as usize]),
                    6 => buf.push(b'\''),
                    7 => buf.push(b' '),
                    8 => buf.extend_from_slice(CHARS[(rng.next_u64() % 12) as usize].as_bytes()),
                    _ => buf.push(rng.next_u64() as u8), // raw byte: invalid UTF-8
                }
            }
            buf.truncate(len);
            // Exactly-sized allocation, so an overrun is a real OOB.
            let exact: Box<[u8]> = buf.into_boxed_slice();
            for scheme in SCHEMES {
                let want = reference_lens(&exact, scheme);
                assert_eq!(
                    walk_lens(&exact, scheme),
                    want,
                    "{scheme:?}: one-at-a-time walk diverged on {:?}",
                    exact
                );
                assert_eq!(
                    fill_lens(&exact, scheme),
                    want,
                    "{scheme:?}: two-phase fill diverged on {:?}",
                    exact
                );
            }
        }
    }
}
