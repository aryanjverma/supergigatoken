use icu::properties::props::{EnumeratedProperty, GeneralCategory, GeneralCategoryGroup, WhiteSpace};
use icu::properties::CodePointSetData;

#[inline]
pub(crate) fn get_general_category(c: char) -> GeneralCategory {
    GeneralCategory::for_char(c)
}

#[inline]
pub(crate) fn is_gc_letter(gc: GeneralCategory) -> bool {
    GeneralCategoryGroup::Letter.contains(gc)
}

#[inline]
pub(crate) fn is_gc_number(gc: GeneralCategory) -> bool {
    GeneralCategoryGroup::Number.contains(gc)
}

/// Unicode White_Space property — matches the same characters as `\s` in regex.
/// This includes GeneralCategory::Separator (Zs/Zl/Zp) PLUS control characters
/// like U+0009 (TAB), U+000A (LF), U+000D (CR), U+0085 (NEL), etc.
#[inline]
pub(crate) fn is_whitespace(c: char) -> bool {
    // The set is a static compiled-data lookup, but cache the borrowed handle
    // to avoid repeated constructor overhead.
    static WS: std::sync::LazyLock<icu::properties::CodePointSetDataBorrowed<'static>> =
        std::sync::LazyLock::new(CodePointSetData::new::<WhiteSpace>);
    WS.contains(c)
}

#[inline]
pub(crate) fn is_letter(c: char) -> bool {
    is_gc_letter(get_general_category(c))
}

#[inline]
pub(crate) fn is_number(c: char) -> bool {
    is_gc_number(get_general_category(c))
}

#[inline]
pub(crate) fn is_other_complete(c: char) -> bool {
    if c.is_ascii() {
        return !c.is_ascii_alphanumeric() && !c.is_ascii_whitespace();
    }
    let gc = get_general_category(c);
    !is_gc_letter(gc) && !is_gc_number(gc) && !is_whitespace(c)
}

// ---------------------------------------------------------------------------
// Packed codepoint → class table (hot-path classification)
// ---------------------------------------------------------------------------

/// Character class as used by the pretokenization regexes: `\p{L}`, `\p{N}`,
/// `\s` (White_Space), and everything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CharClass {
    Letter = 0,
    Number = 1,
    Whitespace = 2,
    Other = 3,
}

/// 2-bit class per codepoint, 4 codepoints per byte (~272 KiB total).
/// A single L1 load replaces the ICU GeneralCategory trie walk plus the
/// White_Space set binary search that the `is_*` predicates above pay per
/// call. Only the cache lines for scripts actually present in the input
/// stay resident.
static CLASS_TABLE: std::sync::LazyLock<Box<[u8]>> =
    std::sync::LazyLock::new(build_class_table);

fn build_class_table() -> Box<[u8]> {
    use icu::properties::CodePointMapData;
    const N: usize = 0x110000;
    let mut classes = vec![CharClass::Other as u8; N];
    let gc = CodePointMapData::<GeneralCategory>::new();
    for (group, class) in [
        (GeneralCategoryGroup::Letter, CharClass::Letter),
        (GeneralCategoryGroup::Number, CharClass::Number),
    ] {
        for range in gc.iter_ranges_for_group(group) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    // White_Space is disjoint from GC Letter/Number, so fill order is moot.
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize].fill(CharClass::Whitespace as u8);
    }
    classes
        .as_chunks::<4>().0.iter()
        .map(|c| c[0] | (c[1] << 2) | (c[2] << 4) | (c[3] << 6))
        .collect()
}

/// Pre-resolved handle to the packed class table. The static is a
/// `LazyLock<Box<[u8]>>`, so every bare [`class_of`] call pays the
/// lazy-init state check plus a dependent load of the Box pointer before
/// the table load itself; per-char classify loops resolve the handle once
/// and index the slice directly.
#[derive(Clone, Copy)]
pub(crate) struct ClassTable(&'static [u8]);

impl ClassTable {
    #[inline]
    pub(crate) fn get() -> Self {
        Self(&CLASS_TABLE)
    }

    /// [`class_of`] without the per-call static resolution. `cp` must be
    /// a valid scalar value (guaranteed when decoded from valid UTF-8).
    #[inline(always)]
    pub(crate) fn class_of(self, cp: u32) -> CharClass {
        debug_assert!(cp < 0x110000);
        // SAFETY: `self.0` is CLASS_TABLE (the only constructor), sized
        // 0x110000 / 4; cp >> 2 is in range for any scalar value.
        let byte = unsafe { *self.0.get_unchecked((cp >> 2) as usize) };
        match (byte >> ((cp & 3) << 1)) & 3 {
            0 => CharClass::Letter,
            1 => CharClass::Number,
            2 => CharClass::Whitespace,
            _ => CharClass::Other,
        }
    }
}

/// Classify a codepoint with one table load. `cp` must be a valid scalar
/// value (guaranteed when decoded from valid UTF-8).
#[inline(always)]
pub(crate) fn class_of(cp: u32) -> CharClass {
    ClassTable::get().class_of(cp)
}

// ---------------------------------------------------------------------------
// DeepSeek character classes (finer split of `Other`)
// ---------------------------------------------------------------------------

/// Character class as used by the DeepSeek V3 main regex, which additionally
/// distinguishes `\p{M}` (joins letter runs) and `\p{P}`/`\p{S}` (punctuation
/// runs) from the remaining `Other` codepoints (controls, format chars,
/// unassigned), which the regex leaves unmatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum DsCharClass {
    Letter = 0,
    Number = 1,
    Whitespace = 2,
    Mark = 3,
    PunctSym = 4,
    Other = 5,
}

/// Four-class view for schemes whose regex joins `\p{M}` into letter
/// runs and excludes it from punctuation runs (Qwen3.5's
/// `[\p{L}\p{M}]+` / `[^\s\p{L}\p{M}\p{N}]+`): marks classify as
/// letters, everything else as in [`class_of`].
#[inline(always)]
pub(crate) fn class_of_marks_join(cp: u32) -> CharClass {
    DsClassTable::get().class_of_marks_join(cp)
}

/// 4-bit class per codepoint, 2 codepoints per byte (~544 KiB total).
static DS_CLASS_TABLE: std::sync::LazyLock<Box<[u8]>> =
    std::sync::LazyLock::new(build_ds_class_table);

fn build_ds_class_table() -> Box<[u8]> {
    use icu::properties::CodePointMapData;
    const N: usize = 0x110000;
    let mut classes = vec![DsCharClass::Other as u8; N];
    let gc = CodePointMapData::<GeneralCategory>::new();
    for (group, class) in [
        (GeneralCategoryGroup::Letter, DsCharClass::Letter),
        (GeneralCategoryGroup::Number, DsCharClass::Number),
        (GeneralCategoryGroup::Mark, DsCharClass::Mark),
        (GeneralCategoryGroup::Punctuation, DsCharClass::PunctSym),
        (GeneralCategoryGroup::Symbol, DsCharClass::PunctSym),
    ] {
        for range in gc.iter_ranges_for_group(group) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    // White_Space is disjoint from the groups above except Zs/Zl/Zp (which
    // are in none of them), so fill order is moot.
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize]
            .fill(DsCharClass::Whitespace as u8);
    }
    classes
        .as_chunks::<2>().0.iter()
        .map(|c| c[0] | (c[1] << 4))
        .collect()
}

/// Pre-resolved handle to the packed DeepSeek class table — same
/// LazyLock-hoist rationale as [`ClassTable`].
#[derive(Clone, Copy)]
pub(crate) struct DsClassTable(&'static [u8]);

impl DsClassTable {
    #[inline]
    pub(crate) fn get() -> Self {
        Self(&DS_CLASS_TABLE)
    }

    /// [`ds_class_of`] without the per-call static resolution. `cp` must
    /// be a valid scalar value (guaranteed when decoded from valid UTF-8).
    #[inline(always)]
    pub(crate) fn ds_class_of(self, cp: u32) -> DsCharClass {
        debug_assert!(cp < 0x110000);
        // SAFETY: `self.0` is DS_CLASS_TABLE (the only constructor), sized
        // 0x110000 / 2; cp >> 1 is in range for any scalar value.
        let byte = unsafe { *self.0.get_unchecked((cp >> 1) as usize) };
        match (byte >> ((cp & 1) << 2)) & 0xF {
            0 => DsCharClass::Letter,
            1 => DsCharClass::Number,
            2 => DsCharClass::Whitespace,
            3 => DsCharClass::Mark,
            4 => DsCharClass::PunctSym,
            _ => DsCharClass::Other,
        }
    }

    /// [`class_of_marks_join`] without the per-call static resolution.
    #[inline(always)]
    pub(crate) fn class_of_marks_join(self, cp: u32) -> CharClass {
        match self.ds_class_of(cp) {
            DsCharClass::Letter | DsCharClass::Mark => CharClass::Letter,
            DsCharClass::Number => CharClass::Number,
            DsCharClass::Whitespace => CharClass::Whitespace,
            DsCharClass::PunctSym | DsCharClass::Other => CharClass::Other,
        }
    }
}

/// Classify a codepoint for the DeepSeek scheme with one table load. `cp`
/// must be a valid scalar value (guaranteed when decoded from valid UTF-8).
#[inline(always)]
pub(crate) fn ds_class_of(cp: u32) -> DsCharClass {
    DsClassTable::get().ds_class_of(cp)
}

// ---------------------------------------------------------------------------
// o200k character classes (case-aware split of Letter)
// ---------------------------------------------------------------------------

/// Character class as used by the o200k regex family (gpt-oss, Nemotron-3),
/// whose letter runs are case-structured:
/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+` etc.
/// `Upper` is Lu|Lt (the strict-uppercase classes that appear only in the
/// first bracket), `Lower` is Ll (only in the second), and `Caseless` is
/// Lm|Lo (in both). Marks (`\p{M}`) are their own class: they join letter
/// runs like `Caseless` but, being outside `\p{L}`, also continue
/// `[^\s\p{L}\p{N}]+` punctuation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum O200kCharClass {
    Upper = 0,
    Lower = 1,
    Caseless = 2,
    Mark = 3,
    Number = 4,
    Whitespace = 5,
    Other = 6,
}

/// 4-bit class per codepoint, 2 codepoints per byte (~544 KiB total).
static O200K_CLASS_TABLE: std::sync::LazyLock<Box<[u8]>> =
    std::sync::LazyLock::new(build_o200k_class_table);

fn build_o200k_class_table() -> Box<[u8]> {
    pack_nibbles(&o200k_classes_unpacked())
}

/// One `O200kCharClass` byte per codepoint (the unpacked form both the
/// o200k and Kimi table builders start from).
fn o200k_classes_unpacked() -> Vec<u8> {
    use icu::properties::CodePointMapData;
    const N: usize = 0x110000;
    let mut classes = vec![O200kCharClass::Other as u8; N];
    let gc = CodePointMapData::<GeneralCategory>::new();
    for (category, class) in [
        (GeneralCategory::UppercaseLetter, O200kCharClass::Upper),
        (GeneralCategory::TitlecaseLetter, O200kCharClass::Upper),
        (GeneralCategory::LowercaseLetter, O200kCharClass::Lower),
        (GeneralCategory::ModifierLetter, O200kCharClass::Caseless),
        (GeneralCategory::OtherLetter, O200kCharClass::Caseless),
    ] {
        for range in gc.iter_ranges_for_value(category) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    for (group, class) in [
        (GeneralCategoryGroup::Mark, O200kCharClass::Mark),
        (GeneralCategoryGroup::Number, O200kCharClass::Number),
    ] {
        for range in gc.iter_ranges_for_group(group) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    // White_Space is disjoint from Letter/Mark/Number, so fill order is moot.
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize]
            .fill(O200kCharClass::Whitespace as u8);
    }
    classes
}

/// Pack one-byte-per-codepoint classes into 4-bit nibbles, 2 per byte.
fn pack_nibbles(classes: &[u8]) -> Box<[u8]> {
    classes
        .as_chunks::<2>().0.iter()
        .map(|c| c[0] | (c[1] << 4))
        .collect()
}

/// Classify a codepoint for the o200k scheme family with one table load.
/// `cp` must be a valid scalar value (guaranteed when decoded from valid
/// UTF-8).
#[inline(always)]
pub(crate) fn o200k_class_of(cp: u32) -> O200kCharClass {
    debug_assert!(cp < 0x110000);
    let byte = unsafe { *O200K_CLASS_TABLE.get_unchecked((cp >> 1) as usize) };
    match (byte >> ((cp & 1) << 2)) & 0xF {
        0 => O200kCharClass::Upper,
        1 => O200kCharClass::Lower,
        2 => O200kCharClass::Caseless,
        3 => O200kCharClass::Mark,
        4 => O200kCharClass::Number,
        5 => O200kCharClass::Whitespace,
        _ => O200kCharClass::Other,
    }
}

// ---------------------------------------------------------------------------
// Kimi character classes (o200k classes with Script=Han split out)
// ---------------------------------------------------------------------------

/// Character class for the Kimi (moonshotai K2 family) regex: the o200k
/// classes with `\p{Han}` split out. The pattern gives Han runs their own
/// leading alternative (`[\p{Han}]+`) and excludes Han from both letter
/// brackets (`[...&&[^\p{Han}]]`), but the general-category rules are
/// otherwise Han-blind: `\p{N}{1,3}` still counts a Han numeral and
/// `[^\s\p{L}\p{N}]+` still spans a Han symbol mid-run. Each Han variant
/// therefore records which base class the char behaves as outside a Han
/// run ([`Self::base`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum KimiCharClass {
    Upper = 0,
    Lower = 1,
    Caseless = 2,
    Mark = 3,
    Number = 4,
    Whitespace = 5,
    Other = 6,
    /// Han letters (Lo, plus Lm like U+3005 々): the bulk of `\p{Han}`.
    /// Never letter-run members; a maximal run of Han-class chars starting
    /// a token is one `[\p{Han}]+` token.
    Han = 7,
    /// Han numerals (Nl: U+3007 〇, Suzhou numerals): `\p{N}` mid-number,
    /// Han-run members otherwise.
    HanNumber = 8,
    /// Han symbols and marks (So: Kangxi radicals; Mc: U+16FF0/1 reading
    /// marks, which `&&[^\p{Han}]` evicts from the letter brackets):
    /// punct-run members mid-run, Han-run members at a token start.
    HanOther = 9,
}

impl KimiCharClass {
    /// The o200k class the char behaves as in non-Han-run contexts.
    #[inline(always)]
    pub(crate) fn base(self) -> O200kCharClass {
        match self {
            KimiCharClass::Upper => O200kCharClass::Upper,
            KimiCharClass::Lower => O200kCharClass::Lower,
            KimiCharClass::Caseless | KimiCharClass::Han => O200kCharClass::Caseless,
            KimiCharClass::Mark => O200kCharClass::Mark,
            KimiCharClass::Number | KimiCharClass::HanNumber => O200kCharClass::Number,
            KimiCharClass::Whitespace => O200kCharClass::Whitespace,
            KimiCharClass::Other | KimiCharClass::HanOther => O200kCharClass::Other,
        }
    }

    /// Is the char in `\p{Han}` (a `[\p{Han}]+` run member)?
    #[inline(always)]
    pub(crate) fn is_han(self) -> bool {
        self as u8 >= KimiCharClass::Han as u8
    }
}

/// 4-bit class per codepoint, 2 codepoints per byte (~544 KiB total).
static KIMI_CLASS_TABLE: std::sync::LazyLock<Box<[u8]>> =
    std::sync::LazyLock::new(build_kimi_class_table);

fn build_kimi_class_table() -> Box<[u8]> {
    use icu::properties::CodePointMapData;
    use icu::properties::props::Script;
    let mut classes = o200k_classes_unpacked();
    let script = CodePointMapData::<Script>::new();
    for range in script.iter_ranges_for_value(Script::Han) {
        for cp in *range.start()..=*range.end() {
            let slot = &mut classes[cp as usize];
            *slot = match *slot {
                c if c == O200kCharClass::Number as u8 => KimiCharClass::HanNumber as u8,
                // Marks land in HanOther: `&&[^\p{Han}]` evicts them from
                // the letter brackets, leaving only their punct-run role.
                c if c == O200kCharClass::Other as u8 || c == O200kCharClass::Mark as u8 => {
                    KimiCharClass::HanOther as u8
                }
                // Letters (Lo/Lm); no Han char is Lu/Lt/Ll/Whitespace.
                _ => KimiCharClass::Han as u8,
            };
        }
    }
    pack_nibbles(&classes)
}

/// Classify a codepoint for the Kimi scheme with one table load. `cp` must
/// be a valid scalar value (guaranteed when decoded from valid UTF-8).
#[inline(always)]
pub(crate) fn kimi_class_of(cp: u32) -> KimiCharClass {
    debug_assert!(cp < 0x110000);
    let byte = unsafe { *KIMI_CLASS_TABLE.get_unchecked((cp >> 1) as usize) };
    match (byte >> ((cp & 1) << 2)) & 0xF {
        0 => KimiCharClass::Upper,
        1 => KimiCharClass::Lower,
        2 => KimiCharClass::Caseless,
        3 => KimiCharClass::Mark,
        4 => KimiCharClass::Number,
        5 => KimiCharClass::Whitespace,
        6 => KimiCharClass::Other,
        7 => KimiCharClass::Han,
        8 => KimiCharClass::HanNumber,
        _ => KimiCharClass::HanOther,
    }
}

/// The CJK ranges isolated by the DeepSeek pretokenizer's second Split:
/// `[\u{4E00}-\u{9FA5}\u{3040}-\u{309F}\u{30A0}-\u{30FF}]` (CJK unified
/// ideographs, hiragana, katakana — the two kana blocks are contiguous).
#[inline(always)]
pub(crate) fn is_deepseek_cjk(cp: u32) -> bool {
    (0x4E00..=0x9FA5).contains(&cp) || (0x3040..=0x30FF).contains(&cp)
}

// ---------------------------------------------------------------------------
// BERT character classes (BertPreTokenizer + BertNormalizer)
// ---------------------------------------------------------------------------
//
// Both tables below are pinned to what `tokenizers` 0.22.2 actually does, not
// to modern Unicode. HF's predicates come from the `unicode_categories` crate,
// whose bundled UCD is several versions behind ICU's compiled data, so "fill
// from ICU" is *not* the same function. Measured by sweeping every scalar
// through `BertPreTokenizer::pre_tokenize_str` and
// `BertNormalizer::normalize_str`, it diverges in three places:
//
//   punctuation      ICU 856  vs  HF 726   -> 141 not punct, 2 extra
//   clean_text       Cc∪Cf∪Co∪Cs           -> 20 kept
//   strip_accents    ICU 2059 Mn vs HF 1567 -> 494 not marks, 2 extra
//
// None of this is academic. U+09FD (Bengali abbreviation sign) and U+0C77
// (Telugu) are live in bert-base-multilingual-cased's languages, and a wrong
// punctuation class shifts the pretoken split, hence the token stream. The mark
// delta is worse: `strip_accents` *deletes*, so treating a codepoint as a mark
// that HF keeps drops it from the text entirely. U+111C9 is the case that
// exposes both at once — `Mn` to ICU and `Po` to HF, so it must be punctuation
// to the splitter and kept by the normalizer.
//
// Deltas MUST be computed against ICU's own sets. Deriving them from a third
// UCD (Python's, say) silently omits everything that UCD leaves unassigned:
// doing so missed 14 punctuation codepoints Unicode 16.0 had newly assigned.
//
// The population assertions in the tests are the guard rail: they pin each
// class's total to the measured HF number, so bumping the `icu` crate turns a
// silent parity divergence into a loud test failure that says "re-measure the
// delta". The end-to-end check is
// `tests/tokenizers/test_wordpiece.py::test_bert_matches_hf_for_every_codepoint`,
// which encodes every scalar through both libraries and compares IDs.

/// Character class for HF's `BertPreTokenizer`, which splits on whitespace
/// (delimiter removed) and then isolates punctuation. A pretoken is therefore
/// either a maximal run of `Word` chars or a single `Punct` char, and
/// `Whitespace` never appears in the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum BertCharClass {
    Word = 0,
    Whitespace = 1,
    Punct = 2,
}

/// `\p{P}` codepoints (per ICU) that HF does **not** classify as punctuation,
/// so `is_bert_punc` returns false and they join word runs. Two causes, one
/// list: codepoints recategorised since HF's UCD snapshot, and codepoints
/// *assigned* since it (Unicode 16.0 additions like U+10D6E GARAY HYPHEN,
/// which HF's tables do not know at all).
///
/// Derived by set-subtracting HF's measured punctuation set from ICU's own
/// `\p{P}` — not from a third UCD's idea of `\p{P}`, which is how the 14
/// Unicode-16 entries were nearly missed: they are unassigned in Python
/// 3.13's UCD 15.1, so a delta computed there does not mention them, and the
/// resulting table over-punctuated by exactly 14 codepoints.
#[rustfmt::skip]
const BERT_STALE_NOT_PUNCT: [u32; 141] = [
    0x061D, 0x09FD, 0x0A76, 0x0C77, 0x0C84,
    0x1B4E, 0x1B4F, 0x1B7D, 0x1B7E, 0x1B7F,
    0x2E43, 0x2E44, 0x2E45, 0x2E46, 0x2E47, 0x2E48, 0x2E49, 0x2E4A, 0x2E4B,
    0x2E4C, 0x2E4D, 0x2E4E, 0x2E4F, 0x2E52, 0x2E53, 0x2E54, 0x2E55, 0x2E56,
    0x2E57, 0x2E58, 0x2E59, 0x2E5A, 0x2E5B, 0x2E5C, 0x2E5D,
    0x10D6E, 0x10EAD, 0x10ED0,
    0x10F55, 0x10F56, 0x10F57, 0x10F58, 0x10F59,
    0x10F86, 0x10F87, 0x10F88, 0x10F89,
    0x113D4, 0x113D5, 0x113D7, 0x113D8,
    0x1144B, 0x1144C, 0x1144D, 0x1144E, 0x1144F, 0x1145A, 0x1145B, 0x1145D,
    0x11660, 0x11661, 0x11662, 0x11663, 0x11664, 0x11665, 0x11666, 0x11667,
    0x11668, 0x11669, 0x1166A, 0x1166B, 0x1166C,
    0x116B9, 0x1183B, 0x11944, 0x11945, 0x11946, 0x119E2,
    0x11A3F, 0x11A40, 0x11A41, 0x11A42, 0x11A43, 0x11A44, 0x11A45, 0x11A46,
    0x11A9A, 0x11A9B, 0x11A9C, 0x11A9E, 0x11A9F, 0x11AA0, 0x11AA1, 0x11AA2,
    0x11B00, 0x11B01, 0x11B02, 0x11B03, 0x11B04, 0x11B05, 0x11B06, 0x11B07,
    0x11B08, 0x11B09, 0x11BE1,
    0x11C41, 0x11C42, 0x11C43, 0x11C44, 0x11C45, 0x11C70, 0x11C71,
    0x11EF7, 0x11EF8,
    0x11F43, 0x11F44, 0x11F45, 0x11F46, 0x11F47, 0x11F48, 0x11F49, 0x11F4A,
    0x11F4B, 0x11F4C, 0x11F4D, 0x11F4E, 0x11F4F,
    0x11FFF, 0x12FF1, 0x12FF2,
    0x16D6D, 0x16D6E, 0x16D6F, 0x16E97, 0x16E98, 0x16E99, 0x16E9A, 0x16FE2,
    0x1E5FF, 0x1E95E, 0x1E95F,
];

/// Codepoints HF treats as punctuation that are outside modern `\p{P}`: both
/// were recategorised after HF's bundled UCD snapshot (U+166D CANADIAN
/// SYLLABICS CHI SIGN Po → So, U+111C9 SHARADA SANDHI MARK Po → Mn).
const BERT_STALE_EXTRA_PUNCT: [u32; 2] = [0x166D, 0x111C9];

/// 2-bit class per codepoint, 4 codepoints per byte (~272 KiB).
static BERT_CLASS_TABLE: std::sync::LazyLock<Box<[u8]>> =
    std::sync::LazyLock::new(build_bert_class_table);

fn build_bert_class_table() -> Box<[u8]> {
    use icu::properties::CodePointMapData;
    const N: usize = 0x110000;
    let mut classes = vec![BertCharClass::Word as u8; N];

    // `is_bert_punc` is `char::is_ascii_punctuation(c) || c.is_punctuation()`.
    let gc = CodePointMapData::<GeneralCategory>::new();
    for range in gc.iter_ranges_for_group(GeneralCategoryGroup::Punctuation) {
        classes[*range.start() as usize..=*range.end() as usize].fill(BertCharClass::Punct as u8);
    }
    // The ASCII half adds `$+<=>^`|~` — Sc/Sm/Sk, not P, but in Rust's
    // `is_ascii_punctuation`.
    for b in 0u8..128 {
        if b.is_ascii_punctuation() {
            classes[b as usize] = BertCharClass::Punct as u8;
        }
    }
    for cp in BERT_STALE_NOT_PUNCT {
        classes[cp as usize] = BertCharClass::Word as u8;
    }
    for cp in BERT_STALE_EXTRA_PUNCT {
        classes[cp as usize] = BertCharClass::Punct as u8;
    }
    // Whitespace last: `str::split(char::is_whitespace)` runs before the
    // punctuation split, so a char that were both would split as whitespace.
    // (White_Space ∩ punct is in fact empty — U+0020 is not
    // `is_ascii_punctuation` — so this only documents the precedence.)
    // Rust's `char::is_whitespace` is exactly the White_Space property, which
    // measured identical to HF's over every scalar: 0 disagreements.
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize]
            .fill(BertCharClass::Whitespace as u8);
    }
    pack_2bit(&classes)
}

fn pack_2bit(classes: &[u8]) -> Box<[u8]> {
    classes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| c[0] | (c[1] << 2) | (c[2] << 4) | (c[3] << 6))
        .collect()
}

/// Byte classes for the walker's scan loops: the ASCII BERT classes plus a
/// [`BERT_BYTE_NON_ASCII`] sentinel for every lead/continuation byte ≥ 0x80.
///
/// Two reasons this is a 256-entry byte table rather than the packed
/// codepoint table. ASCII's classes are fully known at compile time
/// (punctuation is Rust's fixed `is_ascii_punctuation` set, whitespace is
/// `\t\n\v\f\r` and space), so no ICU data is needed; and folding the
/// non-ASCII case into the same table turns "classify a byte, and notice
/// whether it needs UTF-8 decoding" into one load with no separate `b < 0x80`
/// test — which is the entire inner loop of a word-run scan.
pub(crate) const BERT_BYTE_WORD: u8 = 0;
pub(crate) const BERT_BYTE_WS: u8 = 1;
pub(crate) const BERT_BYTE_PUNCT: u8 = 2;
pub(crate) const BERT_BYTE_NON_ASCII: u8 = 3;

pub(crate) const BERT_BYTE_CLASS: [u8; 256] = {
    let mut t = [BERT_BYTE_NON_ASCII; 256];
    let mut b = 0usize;
    while b < 128 {
        t[b] = if (b >= 0x21 && b <= 0x2F)
            || (b >= 0x3A && b <= 0x40)
            || (b >= 0x5B && b <= 0x60)
            || (b >= 0x7B && b <= 0x7E)
        {
            BERT_BYTE_PUNCT
        } else if (b >= 0x09 && b <= 0x0D) || b == 0x20 {
            BERT_BYTE_WS
        } else {
            BERT_BYTE_WORD
        };
        b += 1;
    }
    t
};

/// Pre-resolved handle to the packed BERT class table — same LazyLock-hoist
/// rationale as [`ClassTable`].
#[derive(Clone, Copy)]
pub(crate) struct BertClassTable(&'static [u8]);

impl BertClassTable {
    #[inline]
    pub(crate) fn get() -> Self {
        Self(&BERT_CLASS_TABLE)
    }

    /// [`bert_class_of`] without the per-call static resolution. `cp` must be
    /// a valid scalar value (guaranteed when decoded from valid UTF-8).
    #[inline(always)]
    pub(crate) fn class_of(self, cp: u32) -> BertCharClass {
        debug_assert!(cp < 0x110000);
        // SAFETY: `self.0` is BERT_CLASS_TABLE (the only constructor), sized
        // 0x110000 / 4; cp >> 2 is in range for any scalar value.
        let byte = unsafe { *self.0.get_unchecked((cp >> 2) as usize) };
        match (byte >> ((cp & 3) << 1)) & 3 {
            0 => BertCharClass::Word,
            1 => BertCharClass::Whitespace,
            _ => BertCharClass::Punct,
        }
    }
}

/// Classify a codepoint for `BertPreTokenizer` with one table load. `cp` must
/// be a valid scalar value (guaranteed when decoded from valid UTF-8).
#[inline(always)]
pub(crate) fn bert_class_of(cp: u32) -> BertCharClass {
    BertClassTable::get().class_of(cp)
}

/// Codepoint role in `BertNormalizer`. One load serves both `clean_text`
/// (`Remove` is dropped, `Whitespace` becomes U+0020) and `strip_accents`
/// (`Mark` is dropped after NFD).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum BertNormClass {
    Keep = 0,
    Whitespace = 1,
    Remove = 2,
    Mark = 3,
}

/// `Cf` codepoints (per ICU) that HF's older UCD leaves unassigned, so its
/// control predicate returns false and `clean_text` keeps them. Measured
/// against `tokenizers` 0.22.2; see the module comment.
#[rustfmt::skip]
const BERT_STALE_NOT_CONTROL: [u32; 20] = [
    0x0890, 0x0891, 0x08E2, 0x110CD,
    0x13430, 0x13431, 0x13432, 0x13433, 0x13434, 0x13435, 0x13436, 0x13437,
    0x13438, 0x13439, 0x1343A, 0x1343B, 0x1343C, 0x1343D, 0x1343E, 0x1343F,
];

/// `Mn` codepoints (per ICU) that HF does **not** strip under `strip_accents`,
/// because its older UCD does not classify them as nonspacing marks — most are
/// simply unknown to it (assigned in Unicode 13–16), and U+111C9 was `Po`.
///
/// This is the delta that is easiest to miss, because `strip_accents` is a
/// *deletion*: getting it wrong does not shift a boundary, it silently drops a
/// character. Measured by asking HF whether `"a<c>b"` normalizes to `"ab"` under
/// `strip_accents` alone, for every scalar, and subtracting that from ICU's `Mn`.
/// 494 codepoints; the two in [`BERT_STALE_EXTRA_MARK`] go the other way.
#[rustfmt::skip]
const BERT_STALE_NOT_MARK: [u32; 494] = [
    0x07FD, 0x0897, 0x0898, 0x0899, 0x089A, 0x089B, 0x089C, 0x089D,
    0x089E, 0x089F, 0x08CA, 0x08CB, 0x08CC, 0x08CD, 0x08CE, 0x08CF,
    0x08D0, 0x08D1, 0x08D2, 0x08D3, 0x08D4, 0x08D5, 0x08D6, 0x08D7,
    0x08D8, 0x08D9, 0x08DA, 0x08DB, 0x08DC, 0x08DD, 0x08DE, 0x08DF,
    0x08E0, 0x08E1, 0x09FE, 0x0AFA, 0x0AFB, 0x0AFC, 0x0AFD, 0x0AFE,
    0x0AFF, 0x0B55, 0x0C04, 0x0C3C, 0x0D00, 0x0D3B, 0x0D3C, 0x0D81,
    0x0EBA, 0x0ECE, 0x180F, 0x1885, 0x1886, 0x1ABF, 0x1AC0, 0x1AC1,
    0x1AC2, 0x1AC3, 0x1AC4, 0x1AC5, 0x1AC6, 0x1AC7, 0x1AC8, 0x1AC9,
    0x1ACA, 0x1ACB, 0x1ACC, 0x1ACD, 0x1ACE, 0x1ACF, 0x1AD0, 0x1AD1,
    0x1AD2, 0x1AD3, 0x1AD4, 0x1AD5, 0x1AD6, 0x1AD7, 0x1AD8, 0x1AD9,
    0x1ADA, 0x1ADB, 0x1ADC, 0x1ADD, 0x1AE0, 0x1AE1, 0x1AE2, 0x1AE3,
    0x1AE4, 0x1AE5, 0x1AE6, 0x1AE7, 0x1AE8, 0x1AE9, 0x1AEA, 0x1AEB,
    0x1DF6, 0x1DF7, 0x1DF8, 0x1DF9, 0x1DFA, 0x1DFB, 0xA82C, 0xA8C5,
    0xA8FF, 0xA9BD, 0x10D24, 0x10D25, 0x10D26, 0x10D27, 0x10D69, 0x10D6A,
    0x10D6B, 0x10D6C, 0x10D6D, 0x10EAB, 0x10EAC, 0x10EFA, 0x10EFB, 0x10EFC,
    0x10EFD, 0x10EFE, 0x10EFF, 0x10F46, 0x10F47, 0x10F48, 0x10F49, 0x10F4A,
    0x10F4B, 0x10F4C, 0x10F4D, 0x10F4E, 0x10F4F, 0x10F50, 0x10F82, 0x10F83,
    0x10F84, 0x10F85, 0x11070, 0x11073, 0x11074, 0x110C2, 0x111C9, 0x111CF,
    0x1123E, 0x11241, 0x1133B, 0x113BB, 0x113BC, 0x113BD, 0x113BE, 0x113BF,
    0x113C0, 0x113CE, 0x113D0, 0x113D2, 0x113E1, 0x113E2, 0x11438, 0x11439,
    0x1143A, 0x1143B, 0x1143C, 0x1143D, 0x1143E, 0x1143F, 0x11442, 0x11443,
    0x11444, 0x11446, 0x1145E, 0x1182F, 0x11830, 0x11831, 0x11832, 0x11833,
    0x11834, 0x11835, 0x11836, 0x11837, 0x11839, 0x1183A, 0x1193B, 0x1193C,
    0x1193E, 0x11943, 0x119D4, 0x119D5, 0x119D6, 0x119D7, 0x119DA, 0x119DB,
    0x119E0, 0x11A01, 0x11A02, 0x11A03, 0x11A04, 0x11A05, 0x11A06, 0x11A07,
    0x11A08, 0x11A09, 0x11A0A, 0x11A33, 0x11A34, 0x11A35, 0x11A36, 0x11A37,
    0x11A38, 0x11A3B, 0x11A3C, 0x11A3D, 0x11A3E, 0x11A47, 0x11A51, 0x11A52,
    0x11A53, 0x11A54, 0x11A55, 0x11A56, 0x11A59, 0x11A5A, 0x11A5B, 0x11A8A,
    0x11A8B, 0x11A8C, 0x11A8D, 0x11A8E, 0x11A8F, 0x11A90, 0x11A91, 0x11A92,
    0x11A93, 0x11A94, 0x11A95, 0x11A96, 0x11A98, 0x11A99, 0x11B60, 0x11B62,
    0x11B63, 0x11B64, 0x11B66, 0x11C30, 0x11C31, 0x11C32, 0x11C33, 0x11C34,
    0x11C35, 0x11C36, 0x11C38, 0x11C39, 0x11C3A, 0x11C3B, 0x11C3C, 0x11C3D,
    0x11C3F, 0x11C92, 0x11C93, 0x11C94, 0x11C95, 0x11C96, 0x11C97, 0x11C98,
    0x11C99, 0x11C9A, 0x11C9B, 0x11C9C, 0x11C9D, 0x11C9E, 0x11C9F, 0x11CA0,
    0x11CA1, 0x11CA2, 0x11CA3, 0x11CA4, 0x11CA5, 0x11CA6, 0x11CA7, 0x11CAA,
    0x11CAB, 0x11CAC, 0x11CAD, 0x11CAE, 0x11CAF, 0x11CB0, 0x11CB2, 0x11CB3,
    0x11CB5, 0x11CB6, 0x11D31, 0x11D32, 0x11D33, 0x11D34, 0x11D35, 0x11D36,
    0x11D3A, 0x11D3C, 0x11D3D, 0x11D3F, 0x11D40, 0x11D41, 0x11D42, 0x11D43,
    0x11D44, 0x11D45, 0x11D47, 0x11D90, 0x11D91, 0x11D95, 0x11D97, 0x11EF3,
    0x11EF4, 0x11F00, 0x11F01, 0x11F36, 0x11F37, 0x11F38, 0x11F39, 0x11F3A,
    0x11F40, 0x11F42, 0x11F5A, 0x13440, 0x13447, 0x13448, 0x13449, 0x1344A,
    0x1344B, 0x1344C, 0x1344D, 0x1344E, 0x1344F, 0x13450, 0x13451, 0x13452,
    0x13453, 0x13454, 0x13455, 0x1611E, 0x1611F, 0x16120, 0x16121, 0x16122,
    0x16123, 0x16124, 0x16125, 0x16126, 0x16127, 0x16128, 0x16129, 0x1612D,
    0x1612E, 0x1612F, 0x16F4F, 0x16FE4, 0x1CF00, 0x1CF01, 0x1CF02, 0x1CF03,
    0x1CF04, 0x1CF05, 0x1CF06, 0x1CF07, 0x1CF08, 0x1CF09, 0x1CF0A, 0x1CF0B,
    0x1CF0C, 0x1CF0D, 0x1CF0E, 0x1CF0F, 0x1CF10, 0x1CF11, 0x1CF12, 0x1CF13,
    0x1CF14, 0x1CF15, 0x1CF16, 0x1CF17, 0x1CF18, 0x1CF19, 0x1CF1A, 0x1CF1B,
    0x1CF1C, 0x1CF1D, 0x1CF1E, 0x1CF1F, 0x1CF20, 0x1CF21, 0x1CF22, 0x1CF23,
    0x1CF24, 0x1CF25, 0x1CF26, 0x1CF27, 0x1CF28, 0x1CF29, 0x1CF2A, 0x1CF2B,
    0x1CF2C, 0x1CF2D, 0x1CF30, 0x1CF31, 0x1CF32, 0x1CF33, 0x1CF34, 0x1CF35,
    0x1CF36, 0x1CF37, 0x1CF38, 0x1CF39, 0x1CF3A, 0x1CF3B, 0x1CF3C, 0x1CF3D,
    0x1CF3E, 0x1CF3F, 0x1CF40, 0x1CF41, 0x1CF42, 0x1CF43, 0x1CF44, 0x1CF45,
    0x1CF46, 0x1E000, 0x1E001, 0x1E002, 0x1E003, 0x1E004, 0x1E005, 0x1E006,
    0x1E008, 0x1E009, 0x1E00A, 0x1E00B, 0x1E00C, 0x1E00D, 0x1E00E, 0x1E00F,
    0x1E010, 0x1E011, 0x1E012, 0x1E013, 0x1E014, 0x1E015, 0x1E016, 0x1E017,
    0x1E018, 0x1E01B, 0x1E01C, 0x1E01D, 0x1E01E, 0x1E01F, 0x1E020, 0x1E021,
    0x1E023, 0x1E024, 0x1E026, 0x1E027, 0x1E028, 0x1E029, 0x1E02A, 0x1E08F,
    0x1E130, 0x1E131, 0x1E132, 0x1E133, 0x1E134, 0x1E135, 0x1E136, 0x1E2AE,
    0x1E2EC, 0x1E2ED, 0x1E2EE, 0x1E2EF, 0x1E4EC, 0x1E4ED, 0x1E4EE, 0x1E4EF,
    0x1E5EE, 0x1E5EF, 0x1E6E3, 0x1E6E6, 0x1E6EE, 0x1E6EF, 0x1E6F5, 0x1E944,
    0x1E945, 0x1E946, 0x1E947, 0x1E948, 0x1E949, 0x1E94A,
];

/// Codepoints HF strips as marks that ICU does not classify `Mn`: U+1734
/// (HANUNOO PAMUDPOD, recategorised `Mn` → `Mc` in Unicode 11) and U+1171E
/// (AHOM CONSONANT SIGN MEDIAL RA, `Mn` → `Mc` in Unicode 14).
const BERT_STALE_EXTRA_MARK: [u32; 2] = [0x1734, 0x1171E];

/// 2-bit class per codepoint, 4 codepoints per byte (~272 KiB).
static BERT_NORM_TABLE: std::sync::LazyLock<Box<[u8]>> =
    std::sync::LazyLock::new(build_bert_norm_table);

fn build_bert_norm_table() -> Box<[u8]> {
    use icu::properties::CodePointMapData;
    const N: usize = 0x110000;
    let mut classes = vec![BertNormClass::Keep as u8; N];
    let gc = CodePointMapData::<GeneralCategory>::new();

    // Fill order is load-bearing. HF's `clean_text` first *filters out*
    // controls, then maps the survivors' whitespace to U+0020, so a codepoint
    // that is both (U+0085 NEL, U+000B, U+000C) ends up removed, not spaced.
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize]
            .fill(BertNormClass::Whitespace as u8);
    }
    // HF's control predicate is Cc ∪ Cf ∪ Co ∪ Cs. It must be enumerated
    // category by category: ICU's `GeneralCategoryGroup::Other` also contains
    // Cn (unassigned), and removing those diverges — U+0378 is unassigned and
    // HF keeps it (measured).
    for category in [
        GeneralCategory::Control,
        GeneralCategory::Format,
        GeneralCategory::PrivateUse,
        GeneralCategory::Surrogate,
    ] {
        for range in gc.iter_ranges_for_value(category) {
            classes[*range.start() as usize..=*range.end() as usize]
                .fill(BertNormClass::Remove as u8);
        }
    }
    for cp in BERT_STALE_NOT_CONTROL {
        classes[cp as usize] = BertNormClass::Keep as u8;
    }
    // `\t\n\r` are excepted from the control predicate and are whitespace.
    for cp in [0x09u32, 0x0A, 0x0D] {
        classes[cp as usize] = BertNormClass::Whitespace as u8;
    }
    // The filter also drops NUL (already Cc) and U+FFFD by value — the
    // replacement char is So, so no category fill reaches it.
    classes[0x00] = BertNormClass::Remove as u8;
    classes[0xFFFD] = BertNormClass::Remove as u8;
    // `strip_accents` drops Mn after NFD. Mn is disjoint from White_Space and
    // from the control categories, so this fill cannot collide.
    for range in gc.iter_ranges_for_value(GeneralCategory::NonspacingMark) {
        classes[*range.start() as usize..=*range.end() as usize].fill(BertNormClass::Mark as u8);
    }
    // ... then the staleness delta, which is *not* a rounding error here: 494 of
    // ICU's 2059 Mn codepoints are not marks to HF, and stripping one HF keeps
    // deletes a character outright rather than merely moving a boundary.
    for cp in BERT_STALE_NOT_MARK {
        classes[cp as usize] = BertNormClass::Keep as u8;
    }
    for cp in BERT_STALE_EXTRA_MARK {
        classes[cp as usize] = BertNormClass::Mark as u8;
    }
    pack_2bit(&classes)
}

/// Pre-resolved handle to the packed BERT normalizer table — same
/// LazyLock-hoist rationale as [`ClassTable`].
#[derive(Clone, Copy)]
pub(crate) struct BertNormTable(&'static [u8]);

impl BertNormTable {
    #[inline]
    pub(crate) fn get() -> Self {
        Self(&BERT_NORM_TABLE)
    }

    /// [`bert_norm_class_of`] without the per-call static resolution. `cp`
    /// must be a valid scalar value (guaranteed when decoded from valid
    /// UTF-8).
    #[inline(always)]
    pub(crate) fn class_of(self, cp: u32) -> BertNormClass {
        debug_assert!(cp < 0x110000);
        // SAFETY: `self.0` is BERT_NORM_TABLE (the only constructor), sized
        // 0x110000 / 4; cp >> 2 is in range for any scalar value.
        let byte = unsafe { *self.0.get_unchecked((cp >> 2) as usize) };
        match (byte >> ((cp & 3) << 1)) & 3 {
            0 => BertNormClass::Keep,
            1 => BertNormClass::Whitespace,
            2 => BertNormClass::Remove,
            _ => BertNormClass::Mark,
        }
    }
}

/// Classify a codepoint for `BertNormalizer` with one table load. `cp` must be
/// a valid scalar value (guaranteed when decoded from valid UTF-8).
#[inline(always)]
pub(crate) fn bert_norm_class_of(cp: u32) -> BertNormClass {
    BertNormTable::get().class_of(cp)
}

/// The CJK ranges `BertNormalizer::handle_chinese_chars` pads with spaces.
/// Measured from HF, not copied from a block chart: the ranges are BERT's
/// original hardcoded list, and the U+2B820–U+2B91F hole is real (HF's list
/// jumps from `2A700..=2B81F` straight to `2B920..=2CEAF`, so extension block
/// C's tail is not padded). Kana is **not** included, which is why
/// [`is_deepseek_cjk`] cannot be reused.
#[inline(always)]
pub(crate) fn is_bert_cjk(cp: u32) -> bool {
    // Ordered by likelihood: the main CJK block first, then the BMP extras,
    // then the supplementary planes.
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B81F).contains(&cp)
        || (0x2B920..=0x2CEAF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed table must agree with the ICU predicates for every scalar.
    #[test]
    fn class_table_matches_icu() {
        for cp in 0..=char::MAX as u32 {
            let Some(c) = char::from_u32(cp) else { continue };
            let expected = if is_letter(c) {
                CharClass::Letter
            } else if is_number(c) {
                CharClass::Number
            } else if is_whitespace(c) {
                CharClass::Whitespace
            } else {
                CharClass::Other
            };
            assert_eq!(class_of(cp), expected, "mismatch at U+{cp:04X}");
        }
    }

    /// The o200k table must agree with ICU for every scalar.
    #[test]
    fn o200k_class_table_matches_icu() {
        for cp in 0..=char::MAX as u32 {
            let Some(c) = char::from_u32(cp) else { continue };
            let gc = get_general_category(c);
            let expected = if matches!(
                gc,
                GeneralCategory::UppercaseLetter | GeneralCategory::TitlecaseLetter
            ) {
                O200kCharClass::Upper
            } else if gc == GeneralCategory::LowercaseLetter {
                O200kCharClass::Lower
            } else if is_gc_letter(gc) {
                O200kCharClass::Caseless
            } else if GeneralCategoryGroup::Mark.contains(gc) {
                O200kCharClass::Mark
            } else if is_gc_number(gc) {
                O200kCharClass::Number
            } else if is_whitespace(c) {
                O200kCharClass::Whitespace
            } else {
                O200kCharClass::Other
            };
            assert_eq!(o200k_class_of(cp), expected, "mismatch at U+{cp:04X}");
        }
    }

    /// The Kimi table must refine the o200k table: identical off `\p{Han}`,
    /// and on it a Han variant whose base is the o200k class.
    #[test]
    fn kimi_class_table_refines_o200k() {
        use icu::properties::CodePointMapData;
        use icu::properties::props::Script;
        let script = CodePointMapData::<Script>::new();
        for cp in 0..=char::MAX as u32 {
            let Some(c) = char::from_u32(cp) else { continue };
            let k = kimi_class_of(cp);
            // Han marks base as Other (evicted from the letter brackets, so
            // only their punct-run role remains); all else keeps its class.
            let expected_base = match o200k_class_of(cp) {
                O200kCharClass::Mark if k.is_han() => O200kCharClass::Other,
                c => c,
            };
            assert_eq!(k.base(), expected_base, "base mismatch at U+{cp:04X}");
            assert_eq!(
                k.is_han(),
                script.get(c) == Script::Han,
                "Han mismatch at U+{cp:04X}"
            );
        }
    }

    /// The DeepSeek table must agree with ICU for every scalar, and refine
    /// `class_of` (identical on Letter/Number/Whitespace).
    #[test]
    fn ds_class_table_matches_icu() {
        for cp in 0..=char::MAX as u32 {
            let Some(c) = char::from_u32(cp) else { continue };
            let gc = get_general_category(c);
            let expected = if is_gc_letter(gc) {
                DsCharClass::Letter
            } else if is_gc_number(gc) {
                DsCharClass::Number
            } else if is_whitespace(c) {
                DsCharClass::Whitespace
            } else if GeneralCategoryGroup::Mark.contains(gc) {
                DsCharClass::Mark
            } else if GeneralCategoryGroup::Punctuation.contains(gc)
                || GeneralCategoryGroup::Symbol.contains(gc)
            {
                DsCharClass::PunctSym
            } else {
                DsCharClass::Other
            };
            assert_eq!(ds_class_of(cp), expected, "mismatch at U+{cp:04X}");
        }
    }

    /// The BERT class table must be ICU's `\p{P}` ∪ ASCII punctuation ∪
    /// White_Space, adjusted by exactly the two measured staleness deltas —
    /// nothing else may differ.
    #[test]
    fn bert_class_table_matches_icu_plus_deltas() {
        for cp in 0..=char::MAX as u32 {
            let Some(c) = char::from_u32(cp) else { continue };
            let gc = get_general_category(c);
            let icu_punct = GeneralCategoryGroup::Punctuation.contains(gc)
                || (c.is_ascii() && (c as u8).is_ascii_punctuation());
            let punct = if BERT_STALE_NOT_PUNCT.contains(&cp) {
                false
            } else if BERT_STALE_EXTRA_PUNCT.contains(&cp) {
                true
            } else {
                icu_punct
            };
            let expected = if is_whitespace(c) {
                BertCharClass::Whitespace
            } else if punct {
                BertCharClass::Punct
            } else {
                BertCharClass::Word
            };
            assert_eq!(bert_class_of(cp), expected, "mismatch at U+{cp:04X}");
        }
    }

    /// The normalizer table must be the documented ICU fills adjusted by
    /// exactly the two measured staleness deltas — nothing else may differ.
    #[test]
    fn bert_norm_table_matches_icu_plus_deltas() {
        for cp in 0..=char::MAX as u32 {
            let Some(c) = char::from_u32(cp) else { continue };
            let gc = get_general_category(c);
            let control = matches!(
                gc,
                GeneralCategory::Control
                    | GeneralCategory::Format
                    | GeneralCategory::PrivateUse
                    | GeneralCategory::Surrogate
            ) && !BERT_STALE_NOT_CONTROL.contains(&cp);
            let mark = if BERT_STALE_NOT_MARK.contains(&cp) {
                false
            } else if BERT_STALE_EXTRA_MARK.contains(&cp) {
                true
            } else {
                gc == GeneralCategory::NonspacingMark
            };
            // Precedence, in HF's own order: `\t\n\r` are whitespace; NUL and
            // U+FFFD are removed by value; control beats White_Space; then
            // White_Space; then marks.
            let expected = if matches!(cp, 0x09 | 0x0A | 0x0D) {
                BertNormClass::Whitespace
            } else if matches!(cp, 0x00 | 0xFFFD) {
                BertNormClass::Remove
            } else if control {
                BertNormClass::Remove
            } else if is_whitespace(c) {
                BertNormClass::Whitespace
            } else if mark {
                BertNormClass::Mark
            } else {
                BertNormClass::Keep
            };
            assert_eq!(bert_norm_class_of(cp), expected, "mismatch at U+{cp:04X}");
        }
    }

    /// Class populations, measured by sweeping every scalar through
    /// `tokenizers` 0.22.2 (`BertPreTokenizer::pre_tokenize_str("a" + c + "a")`
    /// and `BertNormalizer::normalize_str("a" + c + "b")`).
    ///
    /// These are the guard rail on the staleness deltas: bumping `icu` to a
    /// newer UCD shifts a count, which is the signal to re-measure the delta
    /// lists against HF rather than to edit the expected number. Surrogates are
    /// excluded because they cannot appear in a `str`, so HF never sees them.
    #[test]
    fn bert_class_populations_match_measured_hf() {
        let mut punct = 0usize;
        let mut ws = 0usize;
        let mut removed = 0usize;
        let mut to_space = 0usize;
        let mut marks = 0usize;
        for cp in 0..=char::MAX as u32 {
            if char::from_u32(cp).is_none() {
                continue;
            }
            match bert_class_of(cp) {
                BertCharClass::Punct => punct += 1,
                BertCharClass::Whitespace => ws += 1,
                BertCharClass::Word => {}
            }
            match bert_norm_class_of(cp) {
                BertNormClass::Remove => removed += 1,
                BertNormClass::Whitespace => to_space += 1,
                BertNormClass::Mark => marks += 1,
                BertNormClass::Keep => {}
            }
        }
        assert_eq!(punct, 726, "punctuation population drifted from HF's");
        assert_eq!(ws, 25, "White_Space population drifted from HF's");
        assert_eq!(removed, 137681, "clean_text removal set drifted from HF's");
        assert_eq!(to_space, 25 - 3, "whitespace-to-space set drifted (\\v\\f are removed)");
        // ICU calls 2059 codepoints Mn; HF strips 1567 of them.
        assert_eq!(marks, 1567, "strip_accents mark set drifted from HF's");
    }

    /// Individually measured codepoints whose classification is
    /// counter-intuitive. Every expectation here came out of the oracle sweep.
    #[test]
    fn bert_class_pins() {
        use BertCharClass::{Punct, Whitespace, Word};
        use BertNormClass::{Keep, Mark, Remove};

        // Unassigned (Cn) is NOT a control for HF — the trap ICU's
        // `GeneralCategoryGroup::Other` would walk straight into.
        assert_eq!(bert_norm_class_of(0x0378), Keep);
        // Both whitespace and control: control wins, so these are removed
        // rather than turned into a space.
        for cp in [0x0085u32, 0x000B, 0x000C] {
            assert_eq!(bert_norm_class_of(cp), Remove, "U+{cp:04X}");
            // ... yet the *pretokenizer* still treats them as whitespace.
            assert_eq!(bert_class_of(cp), Whitespace, "U+{cp:04X}");
        }
        // Controls outside White_Space: removed by the normalizer, and word
        // chars to the pretokenizer (Rust's `is_whitespace` excludes 1C–1F).
        for cp in [0x001Fu32, 0x007F, 0x200B, 0x180E, 0xE000] {
            assert_eq!(bert_norm_class_of(cp), Remove, "U+{cp:04X}");
            assert_eq!(bert_class_of(cp), Word, "U+{cp:04X}");
        }
        // Removed by value, not by category (U+FFFD is So).
        assert_eq!(bert_norm_class_of(0xFFFD), Remove);
        assert_eq!(bert_norm_class_of(0x0000), Remove);
        // Separators become a space.
        for cp in [0x00A0u32, 0x2003, 0x3000, 0x2028, 0x2029] {
            assert_eq!(bert_norm_class_of(cp), BertNormClass::Whitespace, "U+{cp:04X}");
            assert_eq!(bert_class_of(cp), Whitespace, "U+{cp:04X}");
        }
        // Non-ASCII symbols are NOT punctuation to HF, so they join word runs.
        for cp in [0x00D7u32, 0x00F7, 0x20AC, 0x2260, 0x2603, 0x00A6, 0x00B4, 0x02C8] {
            assert_eq!(bert_class_of(cp), Word, "U+{cp:04X}");
        }
        // ... while non-ASCII `\p{P}` is.
        for cp in [0x00A1u32, 0x00AB, 0x2013, 0x2019, 0x3001, 0x00B7] {
            assert_eq!(bert_class_of(cp), Punct, "U+{cp:04X}");
        }
        // Combining marks: word chars for the split, dropped by strip_accents.
        for cp in [0x0301u32, 0x0FBC, 0x0345] {
            assert_eq!(bert_class_of(cp), Word, "U+{cp:04X}");
            assert_eq!(bert_norm_class_of(cp), Mark, "U+{cp:04X}");
        }
        // ... but not every Mn is one to HF. U+111C9 is the sharpest case: `Mn`
        // to ICU and `Po` to HF, so it must be *punctuation* to the splitter and
        // *kept* by strip_accents. Treating it as a mark deleted it silently,
        // which is how this delta was found (a full-codepoint ID sweep against
        // `tokenizers`, not reasoning).
        assert_eq!(bert_class_of(0x111C9), Punct);
        assert_eq!(bert_norm_class_of(0x111C9), Keep);
        for cp in [0x07FDu32, 0x1AC0, 0x11F00, 0x1E944] {
            assert_eq!(bert_norm_class_of(cp), Keep, "U+{cp:04X} is Mn to ICU only");
        }
        // The two that go the other way: Mc to ICU, stripped by HF anyway.
        for cp in [0x1734u32, 0x1171E] {
            assert_eq!(bert_norm_class_of(cp), Mark, "U+{cp:04X}");
        }
        // The staleness deltas themselves.
        assert_eq!(bert_class_of(0x09FD), Word, "Bengali abbreviation sign is Po but pre-dates HF's UCD");
        assert_eq!(bert_class_of(0x2E4F), Word);
        assert_eq!(bert_class_of(0x166D), Punct, "So, yet punctuation to HF");
        assert_eq!(bert_class_of(0x111C9), Punct, "Mn, yet punctuation to HF");
        assert_eq!(bert_norm_class_of(0x0890), Keep, "Cf, yet kept by HF");
        assert_eq!(bert_norm_class_of(0x13430), Keep);
    }

    /// The compile-time byte table must agree with the ICU-built one over
    /// ASCII, and flag everything else as needing a UTF-8 decode.
    #[test]
    fn bert_byte_table_matches_packed() {
        for b in 0u32..128 {
            let expected = match bert_class_of(b) {
                BertCharClass::Word => BERT_BYTE_WORD,
                BertCharClass::Whitespace => BERT_BYTE_WS,
                BertCharClass::Punct => BERT_BYTE_PUNCT,
            };
            assert_eq!(BERT_BYTE_CLASS[b as usize], expected, "U+{b:04X}");
        }
        for b in 128usize..256 {
            assert_eq!(BERT_BYTE_CLASS[b], BERT_BYTE_NON_ASCII, "0x{b:02X}");
        }
    }

    /// `handle_chinese_chars`'s ranges, including the measured U+2B820–2B91F
    /// hole and the absence of kana.
    #[test]
    fn bert_cjk_ranges() {
        for cp in [0x4E00u32, 0x9FFF, 0x3400, 0x4DBF, 0xF900, 0xFAFF, 0x20000, 0x2A6DF, 0x2A700, 0x2B73F, 0x2B740, 0x2B81F, 0x2B920, 0x2CEAF, 0x2F800, 0x2FA1F] {
            assert!(is_bert_cjk(cp), "U+{cp:04X} should be CJK");
        }
        for cp in [0x33FFu32, 0x4DC0, 0xA000, 0xF8FF, 0xFB00, 0x1FFFF, 0x2A6E0, 0x2B820, 0x2B91F, 0x2CEB0, 0x2F7FF, 0x2FA20, 0x3040, 0x30FF] {
            assert!(!is_bert_cjk(cp), "U+{cp:04X} should not be CJK");
        }
        // Kana is CJK to DeepSeek but not to BERT — the reason the two
        // predicates cannot share an implementation.
        assert!(is_deepseek_cjk(0x3040) && !is_bert_cjk(0x3040));
        // U+9FA5 is the top of DeepSeek's range; BERT continues to U+9FFF.
        assert!(is_bert_cjk(0x9FB0) && !is_deepseek_cjk(0x9FB0));
    }
}
