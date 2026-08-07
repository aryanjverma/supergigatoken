//! Two-level encoding for SuperBPE ("superword") tokenizers.
//!
//! A SuperBPE tokenizer is trained in two stages: ordinary whitespace-
//! pretokenized BPE up to a transition point, then a resumed run with the
//! whitespace restriction lifted, learning "superword" tokens that bridge
//! whitespace. At inference the exported tokenizer declares no splitting at
//! all (`ByteLevel { use_regex: false }`, our [`PretokenizerType::Superword`]),
//! so the plain encode path feeds a *whole document* to the byte-level merge
//! loop: the pretoken cache never hits (documents are unique), the SIMD
//! boundary scanners have nothing to find, and the heap runs over thousands
//! of byte symbols per document. That is why SuperBPE encoding measures ~20
//! MB/s against the subword path's GB/s.
//!
//! Almost all of that work is recoverable, because merge priority is the
//! merged token's ID and stage-2 merges are appended after stage-1's. The
//! merge table therefore splits at a token ID `threshold` into
//!
//! - a **prefix** (`id < threshold`) of stage-1 merges, none of which can
//!   span a stage-1 pretoken boundary, and
//! - a **suffix** of superword merges.
//!
//! So one can encode in two levels: split the input at stage-1 pretoken
//! boundaries and encode each piece through the ordinary cached path with
//! the prefix merges (the GB/s path — full cache reuse, seeded vocab, SIMD
//! scan), then run the **full** merge table over the resulting token stream.
//! Level 2 sees ~4.5x fewer symbols than the byte-level path (tokens, not
//! bytes) and the expensive byte-level merging becomes cache hits. Output is
//! bit-identical; `superword_two_level_matches_single_pretoken` checks that
//! against the plain path.
//!
//! # Why the threshold is derivable, and why that is enough
//!
//! `threshold` is not recorded in a `tokenizer.json`, but it does not need
//! to be: it is the smallest merge-result ID whose *junction* — the boundary
//! between its two operands — the stage-1 scheme can place a pretoken
//! boundary on ([`derive_threshold`]). That is the whole safety condition,
//! so correctness does not require the scheme to be the one training
//! actually used; picking the real one is a *performance* matter (a larger
//! prefix means more work on the cached path), so [`CANDIDATE_SCHEMES`] are
//! tried and the largest prefix wins.
//!
//! Testing the junction rather than the whole token matters twice over.
//! Byte-level BPE produces tokens that are fragments of characters — token
//! 6358 of the committed 50k artifact is `b"\xd0\xbe\xd0"`, Cyrillic "о"
//! plus a dangling lead byte — and pretokenizing those describes an input
//! that cannot occur, so a whole-token test mistakes them for superwords and
//! collapses the threshold (6358 instead of 40000). It is also the tighter
//! bound: a junction is all a merge can ever fire on.
//!
//! Four junction shapes really are breakable, and for all of them the answer
//! is to stop splitting there rather than to lower the threshold:
//!
//! - **Inside a whitespace run**, which the `\s+(?!\S)` lookahead can cut. On
//!   the released 128k SuperBPE, 65 sub-threshold merges have whitespace on
//!   both sides of their junction, and a plain stage-1 split diverges from
//!   the reference on 36 of 325 contexts built around them: `"x  y"` encodes
//!   as `"x" | "  " | "y"` but splits as `"x " | " y"`.
//! - **After an apostrophe**, because the contraction alternative makes `'s`
//!   its own pretoken — so `("'", "s")` is merge 424 of the committed
//!   artifact — while a greedy punctuation run swallows the apostrophe in
//!   `"x!'s"` and leaves a boundary between them. Splitting there is a real
//!   divergence, and it dropped the derived threshold from 40000 to 424.
//! - **After a contraction**, the same alternative seen from the other side:
//!   `'s` being its own pretoken puts a boundary between it and the letters
//!   that follow, so `"'st"` splits as `"'s" | "t"` and makes the *tail* of
//!   every contraction a hazard — `("s","t")` is merge **303**, and with it
//!   `("s","e")`, `("t","er")`, `("m","ent")`, `("re","s")`, `("ve","l")`.
//!   This one held the committed artifact's derived threshold at 381 (via
//!   `superbpe_stage1`, below) instead of 40000, so level 1 applied almost no
//!   merges: its output measured 1.73 bytes/symbol — near-raw bytes — and
//!   two-level encoding ran *slower* than the plain path it was meant to beat.
//! - **Between digit runs, and between whitespace and a digit run**, because
//!   `superbpe_stage1`'s digit alternative is a bare `\p{N}{1,3}` with no
//!   leading-space option: `" 1"` splits as `" " | "1"` (merge 381) and
//!   `"0000"` as `"000" | "0"` (merge 939).
//!
//! [`Level1Units`] glues all four, so none of those boundaries survives into
//! level 1. Gluing is safe in the direction that matters — it only ever removes
//! split points, and removing all of them is exactly the plain path — and
//! because [`derive_threshold`] probes the *glued* splitter, the threshold
//! always describes the splitting the encode actually performs. That last
//! clause is load-bearing and no longer self-evident: the probe runs
//! [`Level1Units`] while the encode runs
//! [`crate::pretokenize::fast::level1::Level1Fill`], so the two must agree on
//! every input — including invalid UTF-8, where they would diverge if either
//! walked the scheme's scalar `advance` instead of the mask scanner. Both go
//! through the scanner, and `level1_walkers_agree_*` pins it. A hazard the
//! glue rules do not cover therefore costs speed (a lower threshold, more work
//! in level 2), never correctness. That is a real cost, though, and a silent
//! one: `superword_two_level_matches_single_pretoken` asserts the committed
//! artifact's threshold is exactly its transition point precisely so a new
//! hazard shows up as a failing test rather than as lost throughput.
//!
//! One hazard class is deliberately *not* glued: the o200k-family schemes
//! (`superbpe_stage1`) split case-structured letter runs, so `" Mc" | "C"`
//! (merge 3243), `" You" | "Tube"`, `" Java" | "Script"` are all boundaries.
//! That caps `superbpe_stage1` at 3243 on the committed artifact — which does
//! not matter, because the artifact was trained with `gpt2` and
//! [`SuperwordPlan::build`] takes the best candidate scheme. It *would* matter
//! for a tokenizer genuinely trained with the original stage-1 regex (the
//! released 128k), which needs a camelCase glue rule before its threshold can
//! reach its transition point.

use crate::bpe::tiktoken::Tokenizer;
use crate::pretokenize::fast::level1::glues;
use crate::pretokenize::{FastPretokenizerDispatch, Pretoken, PretokenizerType};
use crate::token::TokenId;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use std::sync::Arc;

/// Stage-1 schemes tried by [`SuperwordPlan::build`]. Both
/// are real SuperBPE stage-1 regexes: `superbpe_stage1` is what the original
/// trainer uses, `gpt2` what `train_superbpe` defaults to.
const CANDIDATE_SCHEMES: [PretokenizerType; 2] =
    [PretokenizerType::SuperBPEStage1, PretokenizerType::GPT2];

/// Contexts [`junction_can_split`] plants a character pair in: one per class
/// the stage-1 regexes distinguish (letter, space, newline, digit, punct,
/// apostrophe for the contraction alternative), plus a space *run* and the
/// empty context, since the whitespace lookahead makes boundary placement
/// depend on more than the immediately adjacent character.
const VERIFY_PAD: [&[u8]; 8] = [b"", b"x", b" ", b"  ", b"\n", b"1", b"!", b"'"];

/// Level-1 splitting for the two-level encode over the *runtime*
/// pretokenizer enum: the stage-1 scheme's pretokens, with adjacent
/// pretokens glued across the four junction shapes a low-ID merge can span
/// (see the module docs for the merge IDs each one otherwise costs) —
///
/// - **whitespace on both sides**, which the `\s+(?!\S)` lookahead can cut
///   mid-run: `"a  b"` yields `"a"` and `"  b"` instead of `"a"`, `" "`,
///   `" b"`;
/// - **a left side ending in an apostrophe**, because the contraction
///   alternative makes `'s` its own pretoken — so `("'", "s")` is a low
///   merge — while a greedy punctuation run swallows the apostrophe in
///   `"x!'s"`, putting a boundary between them. Glued, `"!'"` and `"s"` stay
///   in one unit.
/// - **a left side *starting* with an apostrophe**, the contraction seen from
///   its other end: `"'st"` splits as `"'s"`, `"t"`, so every contraction tail
///   (`s`, `t`, `re`, `ve`, `m`, `ll`, `d`) is a hazardous left operand.
///   Glued, `"'s"` and `"t"` stay together.
/// - **a right side starting with a digit**, since `superbpe_stage1`'s
///   `\p{N}{1,3}` takes no leading space and caps runs at three: `" "`+`"1"`
///   and `"000"`+`"0"` both rejoin.
///
/// All four are rare in prose, so the cache reuse level 1 exists for is intact.
/// No rule is load-bearing on its own: [`derive_threshold`] probes *this*
/// splitter, so a hazard the rules miss lowers the threshold instead of
/// changing anyone's tokens.
///
/// The encode path does not run this walker — it runs
/// [`crate::pretokenize::fast::level1::Level1Fill`], which produces the same
/// units a chunk at a time on the SIMD two-phase fill. This one stays
/// because `derive_threshold`'s scheme is a runtime value, and it is the
/// differential reference the chunked fill is checked against.
pub(crate) struct Level1Units<'a> {
    bytes: &'a [u8],
    inner: FastPretokenizerDispatch<'a>,
    /// Byte offsets of a pretoken pulled from `inner` that did not glue onto
    /// the unit just returned, and so starts the next one.
    pending: Option<(usize, usize)>,
}

impl<'a> Level1Units<'a> {
    pub(crate) fn new(bytes: &'a [u8], scheme: PretokenizerType) -> Self {
        Level1Units {
            bytes,
            inner: scheme.pretokenize(bytes),
            pending: None,
        }
    }

    /// Byte offsets of the next pretoken from `inner`. The pretokenizers yield
    /// consecutive subslices of exactly `self.bytes`, so the address difference
    /// is a valid offset; computed without dereferencing either pointer.
    #[inline(always)]
    fn next_inner(&mut self) -> Option<(usize, usize)> {
        let span = self.inner.next()?.0;
        let start = span.as_ptr().addr() - self.bytes.as_ptr().addr();
        Some((start, start + span.len()))
    }

    /// Byte offsets of the next level-1 unit: the next pretoken extended over
    /// every following pretoken that glues onto it.
    ///
    /// Offsets rather than slices, because that is what
    /// [`crate::pretokenize::fill_spans_keyed_with_buf`] wants: knowing the
    /// backing buffer turns the key pack's long-span branch into a select and
    /// hoists its page-boundary check to one bound per fill.
    #[inline(always)]
    fn next_unit(&mut self) -> Option<(usize, usize)> {
        let (start, mut end) = match self.pending.take() {
            Some(span) => span,
            None => self.next_inner()?,
        };
        // Glue decisions see the pretoken immediately left of the junction, not
        // the whole accumulated unit. The two are interchangeable for a rule
        // that reads the left side's *tail*, but not for one that reads how it
        // starts: `prev` keeps re-satisfying such a rule after every glue, and
        // the unit would swallow the rest of the document.
        let mut prev_start = start;
        while let Some((next_start, next_end)) = self.next_inner() {
            debug_assert_eq!(next_start, end, "pretokens are consecutive");
            if glues(&self.bytes[prev_start..end], &self.bytes[next_start..next_end]) {
                prev_start = next_start;
                end = next_end;
            } else {
                self.pending = Some((next_start, next_end));
                break;
            }
        }
        Some((start, end))
    }
}

impl<'a> Iterator for Level1Units<'a> {
    type Item = Pretoken<'a>;

    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.next_unit()?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

// SAFETY: delegates to `fill_spans_keyed_with_buf`, which writes exactly the
// first `n` entries from spans of `self.bytes`. `next_unit` upholds its
// contract: the offsets come from a pretokenizer over exactly this buffer
// (so `end <= bytes.len()`) and gluing only ever extends a nonempty span
// forward (so `start < end` holds).
unsafe impl<'a> crate::pretokenize::PretokenSpans<'a> for Level1Units<'a> {
    // Out of line for the same reason the mask pretokenizers' impls are: the
    // fill loop and the span walker fuse into one tight function instead of
    // being inlined into the caller's probe loop.
    #[inline(never)]
    fn fill_spans_keyed(
        &mut self,
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        // Copied out first so the closure below can borrow the rest of `self`.
        let bytes = self.bytes;
        crate::pretokenize::fill_spans_keyed_with_buf(
            bytes,
            || self.next_unit(),
            batch,
            prefetch,
        )
    }
}

/// The last character of `a` and the first of `b`, or `None` when the
/// junction between them is not a character boundary.
///
/// Byte-level BPE freely produces tokens that are *fragments* of characters
/// (`b"\xd0\xbe\xd0"` — Cyrillic "о" plus a dangling lead byte — is token
/// 6358 of the committed 50k artifact). A pretoken boundary in valid UTF-8
/// input always falls between whole characters, so a junction that sits
/// inside one can never be a level-1 boundary and the merge is safe by
/// construction. Reporting `None` for those is what keeps fragment tokens
/// from being mistaken for superwords: splitting their bytes with a
/// pretokenizer describes an input that cannot occur.
fn junction_chars(a: &[u8], b: &[u8]) -> Option<(char, char)> {
    let last = (1..=4.min(a.len())).find_map(|k| {
        let mut chars = std::str::from_utf8(&a[a.len() - k..]).ok()?.chars();
        let c = chars.next()?;
        chars.next().is_none().then_some(c)
    })?;
    let first = (1..=4.min(b.len())).find_map(|k| {
        let mut chars = std::str::from_utf8(&b[..k]).ok()?.chars();
        let c = chars.next()?;
        chars.next().is_none().then_some(c)
    })?;
    Some((last, first))
}

/// Whether `scheme` can place a level-1 boundary at the junction between the
/// tokens `left` and `right`.
///
/// Probed rather than reasoned about: the pair is planted in every
/// combination of [`VERIFY_PAD`] contexts and the splitter asked whether a
/// unit ends exactly at the junction. The full token bytes go into the probe,
/// not just the two adjacent characters — a boundary can depend on how the
/// run *started*, which an abstracted pair loses. `"'t"` is a contraction
/// pretoken, so `t|h` is a real boundary in `"'th"`, yet no boundary can
/// follow the `" t"` of `" t" + "he"`; probing characters alone conflates the
/// two and collapses the threshold to 261.
///
/// The pads cover the classes the stage-1 regexes distinguish and include a
/// space *run*, which is what exercises the whitespace lookahead
/// (`\s+(?!\S)`) — the one place a boundary depends on more than the
/// immediate neighbourhood.
///
/// Every junction is probed, with no cheap "same character class must be
/// safe" filter in front. Such a filter looks obviously sound and is not: the
/// o200k-family stage-1 regex has case-structured letter runs, so it splits
/// `camelCase` between two letters, and its `\p{N}{1,3}` splits a digit run
/// every three digits. Leaving the real pretokenizers as the only authority
/// on where boundaries fall is what keeps this honest when a scheme is added.
fn junction_can_split(left: &[u8], right: &[u8], scheme: PretokenizerType) -> bool {
    let mut probe = Vec::with_capacity(left.len() + right.len() + 8);
    for pad_left in VERIFY_PAD {
        for pad_right in VERIFY_PAD {
            probe.clear();
            probe.extend_from_slice(pad_left);
            probe.extend_from_slice(left);
            let junction = probe.len();
            probe.extend_from_slice(right);
            probe.extend_from_slice(pad_right);
            if splits_at(&probe, junction, scheme) {
                return true;
            }
        }
    }
    false
}

/// The smallest merge-result token ID whose junction a level-1 boundary can
/// fall on — the stage-1/stage-2 transition (see the module docs). `None`
/// when no merge can, which is what a plain (non-SuperBPE) tokenizer looks
/// like: every merge stays inside one pretoken, so there is no superword
/// suffix and nothing to recover.
///
/// The threshold is a *safety bound*, not a semantic boundary: any `T` is
/// correct as long as no merge below it can fire at a level-1 junction.
/// Choosing a smaller `T` only moves work from level 1 to level 2 (which
/// carries the full merge table and so still produces every merge), so
/// minimality here is what makes the bound both safe and as large as the
/// junction test can justify.
///
/// Only merge results are considered. Single-byte entries have no junction,
/// and added/special tokens must not be considered at all: contents like
/// `<|endoftext|>` do straddle pretoken boundaries, and a low special token
/// ID (Llama's `<s>` is 1) would otherwise drag the threshold to the bottom
/// of the vocabulary.
pub(crate) fn derive_threshold(
    vocab: &[Arc<[u8]>],
    merges: &HashMap<(TokenId, TokenId), TokenId, FxBuildHasher>,
    scheme: PretokenizerType,
) -> Option<u32> {
    let mut by_id: Vec<(u32, TokenId, TokenId)> = merges
        .iter()
        .map(|(&(left, right), &merged)| (merged.0, left, right))
        .collect();
    by_id.sort_unstable();
    // Ascending IDs mean the first unsafe junction *is* the minimum, so the
    // scan stops at the transition instead of testing the superword suffix.
    for (id, left, right) in by_id {
        let (Some(a), Some(b)) = (vocab.get(left.0 as usize), vocab.get(right.0 as usize)) else {
            continue;
        };
        if junction_chars(a, b).is_some() && junction_can_split(a, b, scheme) {
            return Some(id);
        }
    }
    None
}

/// Whether a level-1 unit boundary falls exactly at byte offset `at`.
fn splits_at(bytes: &[u8], at: usize, scheme: PretokenizerType) -> bool {
    let mut pos = 0;
    for unit in Level1Units::new(bytes, scheme) {
        pos += unit.0.len();
        if pos == at {
            return true;
        }
        if pos > at {
            return false;
        }
    }
    false
}

/// Level-1 boundaries no merge can ever cross, so level 2 can run on the runs
/// *between* them instead of on the whole document.
///
/// # Why the boundaries are derivable
///
/// Take the chronologically **first** merge to span a level-1 boundary `p`,
/// with operands `a` (left) and `b` (right). Nothing has spanned `p` yet, so
/// `a` was built entirely from tokens left of `p` and ends exactly at `p`; the
/// rightmost token it absorbed is therefore the level-1 token `L` immediately
/// left of `p`, and **`a`'s bytes end with `L`'s bytes**. Symmetrically `b`'s
/// bytes start with those of the level-1 token `R` immediately right of `p`.
///
/// So two bit sets decide it: `suffix_left[L]` — some candidate left operand
/// ends with `L` — and `prefix_right[R]`. When either is clear, no merge can
/// ever cross `p`, the two sides are independent, and level 2 may merge them
/// separately. Merging separately is what puts the runs under the tuned
/// small-merge core (`SMALL_MERGE_MAX` = 32) instead of the heap.
///
/// # The candidate set
///
/// Every merge, **except** the ones [`derive_threshold`]'s scan positively
/// cleared: result id below the threshold *and* a junction between two whole
/// characters. Those cannot fire at a level-1 boundary by the threshold
/// argument. Everything else counts — including the sub-threshold merges the
/// scan skipped because their junction sits inside a character
/// ([`junction_chars`] returned `None`), which is what keeps this mechanism
/// from inheriting the scan's "pretoken boundaries fall between whole
/// characters" assumption. Extra candidates only ever *remove* cuts, so a
/// misjudged one costs speed, never correctness.
pub(crate) struct SuperwordCuts {
    /// `suffix_left[t]`: some candidate left operand's bytes end with those of
    /// token `t`. Ids at or above the threshold are set unconditionally — they
    /// only appear mid-merge, where being conservative costs nothing.
    suffix_left: Box<[u64]>,
    /// [`Self::suffix_left`] for candidate right operands and byte prefixes.
    prefix_right: Box<[u64]>,
}

impl SuperwordCuts {
    pub(crate) fn build(
        vocab: &[Arc<[u8]>],
        vocab_inv: &HashMap<Arc<[u8]>, TokenId, FxBuildHasher>,
        merges: &HashMap<(TokenId, TokenId), TokenId, FxBuildHasher>,
        threshold: u32,
    ) -> SuperwordCuts {
        let words = vocab.len().div_ceil(64);
        let mut suffix_left = vec![0u64; words];
        let mut prefix_right = vec![0u64; words];
        // A level-1 token is a sub-threshold vocab entry, so no suffix or
        // prefix longer than the longest of those can ever be one.
        let max_level1_len = vocab[..(threshold as usize).min(vocab.len())]
            .iter()
            .map(|entry| entry.len())
            .max()
            .unwrap_or(0);

        for (&(left, right), &merged) in merges {
            let (Some(a), Some(b)) = (vocab.get(left.0 as usize), vocab.get(right.0 as usize))
            else {
                // Unknown operands: cannot be cleared, so treat as a candidate
                // — but there is nothing to mark from.
                continue;
            };
            if merged.0 < threshold && junction_chars(a, b).is_some() {
                continue;
            }
            // Byte suffixes of `a` / prefixes of `b` that are themselves vocab
            // entries are exactly the tokens that can sit at the junction.
            for k in 1..=a.len().min(max_level1_len) {
                if let Some(&id) = vocab_inv.get(&a[a.len() - k..]) {
                    set_bit(&mut suffix_left, id.0);
                }
            }
            for k in 1..=b.len().min(max_level1_len) {
                if let Some(&id) = vocab_inv.get(&b[..k]) {
                    set_bit(&mut prefix_right, id.0);
                }
            }
        }
        for id in threshold..vocab.len() as u32 {
            set_bit(&mut suffix_left, id);
            set_bit(&mut prefix_right, id);
        }
        SuperwordCuts {
            suffix_left: suffix_left.into_boxed_slice(),
            prefix_right: prefix_right.into_boxed_slice(),
        }
    }

    /// Whether any merge can span the boundary between adjacent level-1 tokens
    /// `left` and `right`. `false` means the boundary is a legal cut.
    ///
    /// Out-of-range ids read as crossable (the conservative answer), so a
    /// vocab/merge inconsistency cannot turn into a wrong token stream.
    #[inline(always)]
    pub(crate) fn crossable(&self, left: u32, right: u32) -> bool {
        get_bit(&self.suffix_left, left) && get_bit(&self.prefix_right, right)
    }

    /// End (exclusive) of the run starting at `start`: the first boundary no
    /// merge can cross, or the end of the stream.
    ///
    /// Called run by run rather than materialised up front. The cut test needs
    /// the *pristine* level-1 tokens at a boundary, and merging run `[s, e)`
    /// writes only inside `[s, e)` and compacts to at or before `s` — so
    /// everything from `e` on is still pristine when the next run is scanned.
    /// Materialising the cut list first cost a second pass over every symbol
    /// plus ~15 MB of pushes per 33 MB of input.
    #[inline(always)]
    pub(crate) fn run_end(&self, symbols: &[u32], start: usize) -> usize {
        for i in start + 1..symbols.len() {
            if !self.crossable(symbols[i - 1], symbols[i]) {
                return i;
            }
        }
        symbols.len()
    }
}

#[inline(always)]
fn set_bit(words: &mut [u64], bit: u32) {
    if let Some(word) = words.get_mut(bit as usize / 64) {
        *word |= 1u64 << (bit % 64);
    }
}

#[inline(always)]
fn get_bit(words: &[u64], bit: u32) -> bool {
    match words.get(bit as usize / 64) {
        Some(word) => word >> (bit % 64) & 1 != 0,
        None => true,
    }
}

/// The two-level encode plan: the stage-1 tokenizer level 1 runs on, plus
/// the scratch buffers the per-unit loop reuses.
pub(crate) struct SuperwordPlan {
    /// Level-1 tokenizer: the same vocab and byte remapping, but only the
    /// merges below the threshold, so its seeded pretoken cache holds
    /// stage-1 encodings. Built once per load and forked per worker.
    pub(crate) stage1: Tokenizer,
    pub(crate) stage1_scheme: PretokenizerType,
    /// Merge-prefix bound: level 1 applies exactly the merges with a merged
    /// ID below this (see [`derive_threshold`]).
    pub(crate) threshold: u32,
    /// Level-2 boundaries no merge can cross ([`SuperwordCuts`]).
    pub(crate) cuts: Arc<SuperwordCuts>,
    /// The unit being encoded: level 1 writes its token ids here and level 2
    /// merges them in place. One buffer, not two — `TokenId` is
    /// `repr(transparent)` over `u32`, so the merge borrows it as
    /// `&mut [TokenId]` instead of copying 7.5 M ids per 33 MB pass.
    pub(crate) symbols: Vec<u32>,
    /// Interleaved-A/B kill switches. The campaign protocol
    /// (`profiling/campaign_report.md` §2) compares variants inside one process
    /// on one set of tables — separate builds differ by more than the effects
    /// being measured, and a second resident tokenizer perturbs the cache — so
    /// both round-2 candidates stay switchable at run time. The branches sit
    /// per segment and per document, never per span or per symbol.
    ///
    /// Defaults come from `GIGATOK_SUPERWORD_L1FILL` /
    /// `GIGATOK_SUPERWORD_NO_CUTS`; `bench_superword_variants` flips them
    /// directly.
    pub(crate) l1_fill: L1Fill,
    pub(crate) use_cuts: bool,
}

/// How level 1 pulls its units — the three shapes
/// `bench_superword_variants` compares (see [`SuperwordPlan::l1_fill`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum L1Fill {
    /// [`Level1Units`] through the generic [`crate::pretokenize::SpanIter`]
    /// fill: an `Iterator::next` per unit on top of the enum-dispatched
    /// `Iterator::next` per stage-1 pretoken.
    Iter,
    /// [`Level1Units`] through
    /// [`crate::pretokenize::fill_spans_keyed_with_buf`]. Removing the
    /// per-span page-boundary branch, the fallible key pack and the per-span
    /// hash-arm dispatch was the obvious win on paper and measured **−1.6%
    /// alone and −5.7% alongside the cuts** (interleaved min-of-5, 33.5 MB
    /// OWT): the buffer-backed helper reaches its CRC arm through a
    /// `#[target_feature(enable = "sse4.2")]` boundary, which turns the glue
    /// walker the closure calls into a real call per span — the same
    /// boundary cost that sank the AVX-512 short-merge scan
    /// (`profiling/x86_port_plan.md` §6). Kept as the measured A/B arm.
    Buf,
    /// [`crate::pretokenize::fast::level1::Level1Spans`]: the mask scanners'
    /// SIMD two-phase fill with the glue rules applied to the harvested
    /// boundary buffer. No iterator and no enum dispatch on the per-pretoken
    /// path, and the same branch-free emission loop the subword path uses —
    /// which is also why it does not pay `Buf`'s cross-`target_feature`
    /// call: the filter is a plain function the fill inlines, not a closure
    /// handed across the boundary.
    ///
    /// The default, and the fastest arm on this box: **+14.3% over `Iter`
    /// alongside the cuts** (156.3 vs 136.7 MB/s, the shipped pairing) and
    /// **+9.7% without them** (129.8 vs 118.3), interleaved min-of-5 over
    /// 33.5 MB of OWT. It pays for that with one extra refill pass per fill
    /// and a one-unit scalar fallback per chunk, both from the deferred last
    /// boundary — priced in, since gluing is why a harvest of `needed`
    /// pretoken boundaries yields fewer than `needed` units.
    TwoPhase,
}

impl SuperwordPlan {
    /// Derive a plan for a tokenizer, or `None` when two-level encoding does
    /// not apply: a plain BPE tokenizer has no merge whose junction can carry
    /// a pretoken boundary, so there is no prefix to split off. (The
    /// remaining exclusions — rank-mapped and `ignore_merges` vocabularies,
    /// whose merge priority is not the token ID — are the caller's, in
    /// `Tokenizer::enable_superword_two_level`.)
    pub(crate) fn build(
        vocab: &Arc<Vec<Arc<[u8]>>>,
        vocab_inv: &HashMap<Arc<[u8]>, TokenId, FxBuildHasher>,
        merges: &HashMap<(TokenId, TokenId), TokenId, FxBuildHasher>,
        byte_remapping: Option<&crate::bpe::ByteRemapping>,
    ) -> Option<SuperwordPlan> {
        Self::build_capped(vocab, vocab_inv, merges, byte_remapping, u32::MAX)
    }

    /// [`Self::build`] with the derived threshold additionally capped at `cap`.
    ///
    /// The cap only ever *lowers* the threshold, which is always safe (it moves
    /// merges from level 1 to level 2, and level 2 carries the full table), so
    /// this cannot express an unsound plan. It exists so
    /// `bench_superword_glue_cost` can measure what a hazard costs *in one
    /// process*: the threshold is a load-time property, so the only way to A/B
    /// it is two plans, and reproducing an old glue rule set by capping is
    /// exact — a hazard's whole effect is the threshold it forces — while
    /// costing the shipped path nothing (no runtime branch in `glues`).
    pub(crate) fn build_capped(
        vocab: &Arc<Vec<Arc<[u8]>>>,
        vocab_inv: &HashMap<Arc<[u8]>, TokenId, FxBuildHasher>,
        merges: &HashMap<(TokenId, TokenId), TokenId, FxBuildHasher>,
        byte_remapping: Option<&crate::bpe::ByteRemapping>,
        cap: u32,
    ) -> Option<SuperwordPlan> {
        let (scheme, threshold) = CANDIDATE_SCHEMES
            .iter()
            .filter_map(|&scheme| Some((scheme, derive_threshold(vocab, merges, scheme)?)))
            // A larger prefix leaves less work for level 2.
            .max_by_key(|&(_, threshold)| threshold)
            .map(|(scheme, threshold)| (scheme, threshold.min(cap)))?;

        let stage1_merges: HashMap<(TokenId, TokenId), TokenId, FxBuildHasher> = merges
            .iter()
            .filter(|(_, merged)| merged.0 < threshold)
            .map(|(&pair, &merged)| (pair, merged))
            .collect();
        let mut stage1 = Tokenizer::new(
            stage1_merges,
            vocab.iter().map(|entry| entry.to_vec()).collect(),
            byte_remapping.cloned(),
        );
        // Level 1 is fed `Level1Units` explicitly rather than through this
        // tokenizer's own pipeline; the scheme is recorded so `Debug` and
        // any future direct use agree with what the plan actually splits on.
        stage1.set_pretokenizer_type(scheme);
        Some(SuperwordPlan {
            stage1,
            stage1_scheme: scheme,
            threshold,
            cuts: Arc::new(SuperwordCuts::build(vocab, vocab_inv, merges, threshold)),
            symbols: Vec::new(),
            l1_fill: env_l1_fill(),
            use_cuts: !env_flag("GIGATOK_SUPERWORD_NO_CUTS"),
        })
    }

    /// A plan sharing the same model tables with a freshly seeded level-1
    /// cache, for per-worker forks (see `Tokenizer::fork_sized`).
    pub(crate) fn fork_sized(&self, expected_bytes: usize) -> SuperwordPlan {
        SuperwordPlan {
            stage1: self.stage1.fork_sized(expected_bytes),
            stage1_scheme: self.stage1_scheme,
            threshold: self.threshold,
            cuts: Arc::clone(&self.cuts),
            symbols: Vec::new(),
            l1_fill: self.l1_fill,
            use_cuts: self.use_cuts,
        }
    }
}

/// Whether an environment variable is set to anything but `0`.
fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| v != "0")
}

/// `GIGATOK_SUPERWORD_L1FILL`: `iter`, `buf`, or `twophase` (the default,
/// and anything unrecognised).
fn env_l1_fill() -> L1Fill {
    match std::env::var_os("GIGATOK_SUPERWORD_L1FILL") {
        Some(v) if v.eq_ignore_ascii_case("iter") => L1Fill::Iter,
        Some(v) if v.eq_ignore_ascii_case("buf") => L1Fill::Buf,
        _ => L1Fill::TwoPhase,
    }
}

/// Test hooks for `boundary_whitespace_matches_decode`, which compares the
/// ASCII fast lanes against the decode they short-circuit.
#[cfg(test)]
pub(crate) fn ends_with_whitespace_for_test(s: &[u8]) -> bool {
    crate::pretokenize::fast::level1::ends_with_whitespace(s)
}

#[cfg(test)]
pub(crate) fn starts_with_whitespace_for_test(s: &[u8]) -> bool {
    crate::pretokenize::fast::level1::starts_with_whitespace(s)
}

