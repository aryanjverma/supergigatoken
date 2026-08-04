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
//! Two junction shapes really are breakable, and for both the answer is to
//! stop splitting there rather than to lower the threshold:
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
//!
//! [`Level1Units`] glues both, so neither boundary survives into level 1.
//! Gluing is safe in the direction that matters — it only ever removes split
//! points, and removing all of them is exactly the plain path — and because
//! [`derive_threshold`] probes the *glued* splitter, the threshold always
//! describes the splitting the encode actually performs. A hazard the glue
//! rules do not cover therefore costs speed (a lower threshold, more work in
//! level 2), never correctness.

use crate::bpe::tiktoken::Tokenizer;
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

/// Level-1 splitting for the two-level encode: the stage-1 scheme's
/// pretokens, with adjacent pretokens glued across the two junction shapes a
/// low-ID merge can span (see the module docs) —
///
/// - **whitespace on both sides**, which the `\s+(?!\S)` lookahead can cut
///   mid-run: `"a  b"` yields `"a"` and `"  b"` instead of `"a"`, `" "`,
///   `" b"`;
/// - **a left side ending in an apostrophe**, because the contraction
///   alternative makes `'s` its own pretoken — so `("'", "s")` is a low
///   merge — while a greedy punctuation run swallows the apostrophe in
///   `"x!'s"`, putting a boundary between them. Glued, `"!'"` and `"s"` stay
///   in one unit.
///
/// Both are rare in prose, so the cache reuse level 1 exists for is intact.
/// Neither rule is load-bearing on its own: [`derive_threshold`] probes *this*
/// splitter, so a hazard the rules miss lowers the threshold instead of
/// changing anyone's tokens.
pub(crate) struct Level1Units<'a> {
    bytes: &'a [u8],
    inner: FastPretokenizerDispatch<'a>,
    /// A pretoken pulled from `inner` that did not glue onto the unit just
    /// returned, and so starts the next one.
    pending: Option<&'a [u8]>,
}

impl<'a> Level1Units<'a> {
    pub(crate) fn new(bytes: &'a [u8], scheme: PretokenizerType) -> Self {
        Level1Units {
            bytes,
            inner: scheme.pretokenize(bytes),
            pending: None,
        }
    }

    /// Byte offset of `span` within `self.bytes`. The pretokenizers yield
    /// consecutive subslices of exactly this buffer, so the difference is a
    /// valid offset; computed on addresses, without dereferencing either
    /// pointer.
    #[inline]
    fn offset_of(&self, span: &[u8]) -> usize {
        span.as_ptr().addr() - self.bytes.as_ptr().addr()
    }
}

impl<'a> Iterator for Level1Units<'a> {
    type Item = Pretoken<'a>;

    fn next(&mut self) -> Option<Pretoken<'a>> {
        let mut cur = match self.pending.take() {
            Some(span) => span,
            None => self.inner.next()?.0,
        };
        while let Some(next) = self.inner.next() {
            if (ends_with_whitespace(cur) && starts_with_whitespace(next.0)) || cur.ends_with(b"'")
            {
                // Glue: `next` directly follows `cur` in `self.bytes`.
                let start = self.offset_of(cur);
                let end = self.offset_of(next.0) + next.0.len();
                debug_assert_eq!(start + cur.len(), self.offset_of(next.0));
                cur = &self.bytes[start..end];
            } else {
                self.pending = Some(next.0);
                break;
            }
        }
        Some(Pretoken(cur))
    }
}

/// Decode the last character of `s` and report whether it is whitespace.
///
/// `s` is a pretoken, so it is non-empty, but it need not be valid UTF-8
/// (the pretokenizers pass malformed bytes through). An undecodable tail
/// counts as whitespace, which only makes [`Level1Units`] glue more — the
/// safe direction.
fn ends_with_whitespace(s: &[u8]) -> bool {
    for k in 1..=4.min(s.len()) {
        if let Ok(tail) = std::str::from_utf8(&s[s.len() - k..]) {
            return tail.chars().next_back().is_some_and(char::is_whitespace);
        }
    }
    true
}

/// [`ends_with_whitespace`] for the first character of `s`.
fn starts_with_whitespace(s: &[u8]) -> bool {
    for k in 1..=4.min(s.len()) {
        if let Ok(head) = std::str::from_utf8(&s[..k]) {
            return head.chars().next().is_some_and(char::is_whitespace);
        }
    }
    true
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
    /// Level-1 token ids for the unit being encoded.
    pub(crate) level1: Vec<u32>,
    /// `level1` as `TokenId`s, which the level-2 merge loop merges in place.
    pub(crate) symbols: Vec<TokenId>,
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
        merges: &HashMap<(TokenId, TokenId), TokenId, FxBuildHasher>,
        byte_remapping: Option<&crate::bpe::ByteRemapping>,
    ) -> Option<SuperwordPlan> {
        let (scheme, threshold) = CANDIDATE_SCHEMES
            .iter()
            .filter_map(|&scheme| Some((scheme, derive_threshold(vocab, merges, scheme)?)))
            // A larger prefix leaves less work for level 2.
            .max_by_key(|&(_, threshold)| threshold)?;

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
            level1: Vec::new(),
            symbols: Vec::new(),
        })
    }

    /// A plan sharing the same model tables with a freshly seeded level-1
    /// cache, for per-worker forks (see `Tokenizer::fork_sized`).
    pub(crate) fn fork_sized(&self, expected_bytes: usize) -> SuperwordPlan {
        SuperwordPlan {
            stage1: self.stage1.fork_sized(expected_bytes),
            stage1_scheme: self.stage1_scheme,
            threshold: self.threshold,
            level1: Vec::new(),
            symbols: Vec::new(),
        }
    }
}
