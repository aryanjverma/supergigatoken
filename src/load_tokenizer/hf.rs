//! Load HuggingFace tokenizer.json files.
//!
//! Supports two styles:
//! - SentencePiece BPE with `byte_fallback=true` (e.g. Llama) → [`load_hf_sentencepiece`]
//! - ByteLevel BPE without byte_fallback (e.g. GPT-2) → [`load_hf_bpe`]

// The tokenizer variants differ greatly in size
#![allow(clippy::large_enum_variant)]

use crate::bpe::sentencepiece::{AddedTokenSpec, Metaspace, NormOp, PrependScheme};
use crate::bpe::{self, SentencePieceBPE};
use crate::token::TokenId;
use eyre::{Context, Result, ensure};
use rustc_hash::FxBuildHasher;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// JSON schema (only the fields we need)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TokenizerJson {
    model: Model,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    #[serde(default)]
    pre_tokenizer: Option<PreTokenizerJson>,
    #[serde(default)]
    normalizer: Option<NormalizerJson>,
    #[serde(default)]
    decoder: Option<DecoderJson>,
}

/// The `decoder` block. Only the `WordPiece` shape is modelled: the byte-level
/// decoders describe an inverse the byte-level path already performs by
/// construction, and a WordPiece file's decoder is the only one whose absence
/// or settings change the decoded string (see
/// [`crate::bpe::wordpiece::WordPieceDecoder`]).
#[derive(Deserialize)]
struct DecoderJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    decoders: Vec<DecoderJson>,
    #[serde(default)]
    prefix: Option<String>,
    #[serde(default)]
    cleanup: Option<bool>,
}

#[derive(Deserialize)]
struct NormalizerJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    normalizers: Vec<NormalizerJson>,
    /// `Prepend` normalizer: the prefix (Llama 2's "▁").
    #[serde(default)]
    prepend: Option<String>,
    /// `Replace` normalizer: pattern and replacement content.
    #[serde(default)]
    pattern: Option<PatternJson>,
    #[serde(default)]
    content: Option<String>,
    /// `Strip` normalizer sides.
    #[serde(default)]
    strip_left: Option<bool>,
    #[serde(default)]
    strip_right: Option<bool>,
    /// `Precompiled` normalizer: base64-encoded sentencepiece charsmap.
    #[serde(default)]
    precompiled_charsmap: Option<String>,
    /// `BertNormalizer` flags. `strip_accents` is a tri-state: `null` means
    /// "follow `lowercase`", which is not the same as `false`.
    #[serde(default)]
    clean_text: Option<bool>,
    #[serde(default)]
    handle_chinese_chars: Option<bool>,
    #[serde(default)]
    strip_accents: Option<bool>,
    #[serde(default)]
    lowercase: Option<bool>,
}

#[derive(Deserialize)]
struct PreTokenizerJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    pretokenizers: Vec<PreTokenizerJson>,
    #[serde(default)]
    pattern: Option<PatternJson>,
    /// `Metaspace` fields. `add_prefix_space` is the pre-0.15 spelling of
    /// `prepend_scheme`.
    #[serde(default)]
    replacement: Option<String>,
    #[serde(default)]
    prepend_scheme: Option<String>,
    #[serde(default)]
    add_prefix_space: Option<bool>,
    /// `ByteLevel` field: when `false`, the scheme does no regex splitting at
    /// all — the whitespace-lifted "superword" scheme SuperBPE uses.
    #[serde(default)]
    use_regex: Option<bool>,
    #[serde(default)]
    split: Option<bool>,
    /// `Split` field (e.g. "MergedWithPrevious" for the gemma-3/4 no-op
    /// space Split).
    #[serde(default)]
    behavior: Option<String>,
}

#[derive(Deserialize)]
struct PatternJson {
    #[serde(rename = "Regex", default)]
    regex: Option<String>,
    #[serde(rename = "String", default)]
    literal: Option<String>,
}

#[derive(Deserialize)]
struct Model {
    /// tokenizer.json files written before tokenizers 0.9 (e.g. the original
    /// GPT-2 upload) omit `model.type`; those are always BPE.
    #[serde(rename = "type", default = "legacy_bpe_type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    /// Absent on WordPiece models, which have no merge list at all.
    #[serde(default, deserialize_with = "deserialize_merges")]
    merges: Vec<[String; 2]>,
    #[serde(default)]
    byte_fallback: bool,
    /// WordPiece: the token every failed segmentation collapses to.
    #[serde(default)]
    unk_token: Option<String>,
    /// WordPiece: the prefix marking a non-word-initial piece (`"##"`).
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    /// WordPiece: word length cap **in chars**, past which the word becomes a
    /// single unk. Also the untyped-legacy WordPiece marker.
    #[serde(default)]
    max_input_chars_per_word: Option<usize>,
    /// HF BPE `ignore_merges`: a pretoken whose whole byte string is a vocab
    /// entry encodes as that single ID, skipping the merge loop (GLM-5.2,
    /// DeepSeek V3, Llama 3).
    #[serde(default)]
    ignore_merges: bool,
}

fn legacy_bpe_type() -> String {
    "BPE".to_string()
}

/// Merges appear as `["a", "b"]` arrays in current tokenizer.json files and
/// as `"a b"` strings in older ones; accept both.
fn deserialize_merges<'de, D>(deserializer: D) -> Result<Vec<[String; 2]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Merge {
        Pair([String; 2]),
        Legacy(String),
    }
    let raw = Vec::<Merge>::deserialize(deserializer)?;
    raw.into_iter()
        .map(|m| match m {
            Merge::Pair(pair) => Ok(pair),
            Merge::Legacy(s) => {
                let (a, b) = s.split_once(' ').ok_or_else(|| {
                    serde::de::Error::custom(format!("invalid merge entry: {s:?}"))
                })?;
                Ok([a.to_string(), b.to_string()])
            }
        })
        .collect()
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
}

// ---------------------------------------------------------------------------
// Token string → raw bytes conversion
// ---------------------------------------------------------------------------

/// Parse a byte-fallback token string `<0xHH>` into its byte.
fn parse_byte_fallback(s: &str) -> Option<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}

/// Convert a HuggingFace vocab string to raw bytes.
///
/// - Byte-fallback tokens `<0xHH>` → the single byte.
/// - Everything else → its UTF-8 bytes (▁ is kept as-is).
fn token_str_to_bytes(s: &str) -> Vec<u8> {
    match parse_byte_fallback(s) {
        Some(byte) => vec![byte],
        None => s.as_bytes().to_vec(),
    }
}

/// Added tokens may live outside model.vocab (e.g. Qwen2's <|endoftext|>,
/// Phi-3's placeholders); extend the vocab so their IDs decode to the
/// literal content.
fn extend_vocab_with_added_tokens(vocab: &mut Vec<Arc<[u8]>>, added_tokens: &[AddedToken]) {
    for t in added_tokens {
        let id = t.id as usize;
        if id >= vocab.len() {
            vocab.resize(id + 1, Arc::from(Vec::new().as_slice()));
        }
        if vocab[id].is_empty() {
            vocab[id] = t.content.as_bytes().into();
        }
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// A tokenizer loaded from HuggingFace `tokenizer.json` data: the model's
/// `byte_fallback` flag decides which of the two supported styles applies.
pub enum HfTokenizer {
    Bpe(bpe::tiktoken::Tokenizer),
    SentencePiece(SentencePieceBPE),
}

/// Probes `model.type` alone, so an unsupported model family (Unigram, ...) is
/// refused by name BEFORE the full BPE-shaped schema is applied — those files
/// are valid JSON with a different `model.vocab`/`merges` shape, and the full
/// parse would report a misleading deserializer error ("invalid type: sequence,
/// expected a map").
#[derive(Deserialize)]
struct ModelTypeProbe {
    #[serde(default)]
    model: Option<ModelTypeOnly>,
}

#[derive(Deserialize)]
struct ModelTypeOnly {
    #[serde(rename = "type")]
    model_type: Option<String>,
    /// Family markers for untyped legacy files (pre-0.9 `tokenizers`
    /// omitted `model.type`): `unk_id` only exists on Unigram models
    /// (e.g. t5-small, xlm-roberta) and `max_input_chars_per_word` only on
    /// WordPiece (e.g. bert-base-uncased, whose `model` block has no `type`).
    /// `continuing_subword_prefix` would NOT work for WordPiece detection:
    /// BPE serializes it too (the original gpt2 upload has
    /// `"continuing_subword_prefix": ""`).
    unk_id: Option<u64>,
    max_input_chars_per_word: Option<u64>,
}

/// Which model family a `tokenizer.json` declares.
#[derive(PartialEq, Eq, Debug)]
enum ModelFamily {
    Bpe,
    WordPiece,
}

/// The family, or an error naming an unsupported one. Typed files say so
/// outright; untyped legacy files are identified by their marker fields.
fn probe_model_family(data: &[u8]) -> Result<ModelFamily> {
    let Ok(ModelTypeProbe { model: Some(m) }) = sonic_rs::from_slice::<ModelTypeProbe>(data) else {
        // No `model` object to probe; let the full parse produce the error.
        return Ok(ModelFamily::Bpe);
    };
    let unsupported = match m.model_type.as_deref() {
        Some("BPE") => return Ok(ModelFamily::Bpe),
        Some("WordPiece") => return Ok(ModelFamily::WordPiece),
        Some(other) => other.to_string(),
        None if m.unk_id.is_some() => "Unigram (untyped legacy file)".to_string(),
        None if m.max_input_chars_per_word.is_some() => return Ok(ModelFamily::WordPiece),
        // Untyped BPE (pre-0.9 GPT-2-style files).
        None => return Ok(ModelFamily::Bpe),
    };
    Err(eyre::eyre!(
        "Unsupported model type \"{unsupported}\": gigatoken supports BPE tokenizers \
         (byte-level, or SentencePiece-style with byte_fallback) and WordPiece"
    ))
}

fn parse_tokenizer_json(data: &[u8]) -> Result<TokenizerJson> {
    probe_model_family(data)?;
    // Inline the deserializer's own message (offending field, position,
    // snippet): the first line is often all that surfaces in test summaries
    // and short tracebacks.
    sonic_rs::from_slice(data).map_err(|e| eyre::eyre!("Failed to parse tokenizer JSON: {e}"))
}

fn read_tokenizer_json(path: impl AsRef<Path>) -> Result<TokenizerJson> {
    let path = path.as_ref();
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    parse_tokenizer_json(&data).with_context(|| format!("Failed to parse {}", path.display()))
}

/// Load a tokenizer from in-memory `tokenizer.json` contents, choosing the
/// SentencePiece or ByteLevel BPE style from the model's `byte_fallback` flag.
pub fn load_hf_slice(data: &[u8]) -> Result<HfTokenizer> {
    let family = probe_model_family(data)?;
    let tj = parse_tokenizer_json(data)?;
    // WordPiece rides the `Bpe` arm: `build_wordpiece` returns the same
    // `Tokenizer` type with its WordPiece model installed, so no enum variant,
    // no third `batch.rs` function family, and no Python dispatch change.
    if family == ModelFamily::WordPiece {
        Ok(HfTokenizer::Bpe(build_wordpiece(&tj)?))
    } else if tj.model.byte_fallback {
        Ok(HfTokenizer::SentencePiece(build_sentencepiece(&tj)?))
    } else {
        Ok(HfTokenizer::Bpe(build_bpe(&tj)?))
    }
}

/// Load a HuggingFace `tokenizer.json` that uses SentencePiece-style BPE with
/// byte fallback (e.g. Llama 2 / TinyLlama).
///
/// Returns a [`SentencePieceBPE`] that preserves the original HF token IDs.
pub fn load_hf_sentencepiece(path: impl AsRef<Path>) -> Result<SentencePieceBPE> {
    build_sentencepiece(&read_tokenizer_json(path)?)
}

fn build_sentencepiece(tj: &TokenizerJson) -> Result<SentencePieceBPE> {
    ensure!(
        tj.model.model_type == "BPE",
        "Unsupported model type: {} (expected BPE)",
        tj.model.model_type
    );
    ensure!(
        tj.model.byte_fallback,
        "Only byte_fallback tokenizers are supported"
    );

    let hf_vocab = &tj.model.vocab;
    let hf_merges = &tj.model.merges;

    // --- Build vocab (preserving original HF IDs) ----------------------------

    let max_id = hf_vocab.values().max().copied().unwrap_or(0) as usize;
    let mut vocab: Vec<Arc<[u8]>> = vec![Arc::from(Vec::new().as_slice()); max_id + 1];
    let mut vocab_inv: HashMap<Arc<[u8]>, TokenId, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(hf_vocab.len(), FxBuildHasher);

    // Insert byte-fallback tokens first, then character tokens, so that
    // character tokens win in vocab_inv when both map to the same bytes.
    let mut byte_fallback_entries = Vec::new();
    let mut other_entries = Vec::new();
    for (tok_str, &id) in hf_vocab {
        if parse_byte_fallback(tok_str).is_some() {
            byte_fallback_entries.push((tok_str, id));
        } else {
            other_entries.push((tok_str, id));
        }
    }
    for (tok_str, id) in byte_fallback_entries {
        let bytes: Arc<[u8]> = token_str_to_bytes(tok_str).into();
        vocab[id as usize] = bytes.clone();
        vocab_inv.insert(bytes, TokenId::from(id));
    }
    for (tok_str, id) in other_entries {
        let bytes: Arc<[u8]> = token_str_to_bytes(tok_str).into();
        vocab[id as usize] = bytes.clone();
        vocab_inv.insert(bytes, TokenId::from(id));
    }

    // --- Extract byte-fallback token IDs -------------------------------------

    // Some vocabs omit byte tokens they never need (Gemma has literal `\t`
    // pieces instead of `<0x09>`); those stay `None`.
    let mut byte_fallback_ids = [None; 256];
    for byte_val in 0u16..=255 {
        let key = format!("<0x{:02X}>", byte_val);
        byte_fallback_ids[byte_val as usize] = hf_vocab.get(&key).map(|&id| TokenId::from(id));
    }
    ensure!(
        byte_fallback_ids.iter().any(|id| id.is_some()),
        "byte_fallback is set but the vocab has no <0xHH> byte tokens"
    );

    // --- Build merge table (with explicit ranks) -----------------------------

    let mut merges: HashMap<u64, (TokenId, u32), FxBuildHasher> =
        HashMap::with_capacity_and_hasher(hf_merges.len(), FxBuildHasher);

    let hf_str_to_id = |s: &str| -> Option<TokenId> {
        let bytes = token_str_to_bytes(s);
        vocab_inv.get(bytes.as_slice()).copied()
    };

    for (rank, [str_a, str_b]) in hf_merges.iter().enumerate() {
        let id_a = match hf_str_to_id(str_a) {
            Some(id) => id,
            None => continue,
        };
        let id_b = match hf_str_to_id(str_b) {
            Some(id) => id,
            None => continue,
        };

        let merged_str = format!("{str_a}{str_b}");
        let id_merged = match hf_str_to_id(&merged_str) {
            Some(id) => id,
            None => continue,
        };

        merges
            .entry(crate::bpe::ranked_merge_key(id_a, id_b))
            .or_insert((id_merged, rank as u32));
    }

    // --- Normalizer and pre-tokenizer configuration --------------------------

    let mut norm_ops = Vec::new();
    if let Some(n) = &tj.normalizer {
        parse_sp_normalizer(n, &mut norm_ops)?;
    }
    let metaspace = parse_sp_metaspace(&tj.pre_tokenizer, &norm_ops)?;

    // --- Extract added tokens (for splitting before encoding) ----------------

    // All added tokens (special and non-special) are matched atomically by
    // HF's AddedVocabulary; mirror that. `normalized: false` tokens match in
    // the raw input, `normalized: true` ones match against normalizer output
    // with their content normalized the same way.
    let mut added_tokens = Vec::new();
    let mut norm_added_tokens = Vec::new();
    for t in &tj.added_tokens {
        let spec = AddedTokenSpec {
            content: t.content.clone(),
            id: TokenId::from(t.id),
            lstrip: t.lstrip,
            rstrip: t.rstrip,
        };
        if t.normalized {
            norm_added_tokens.push(spec);
        } else {
            added_tokens.push(spec);
        }
    }
    extend_vocab_with_added_tokens(&mut vocab, &tj.added_tokens);

    let mut model = SentencePieceBPE {
        merges,
        vocab,
        vocab_inv,
        byte_fallback_ids,
        added_tokens,
        norm_added_tokens: Vec::new(),
        norm_ops,
        metaspace,
        word_split: crate::bpe::sentencepiece::WordSplit::None,
        raw_prepend: None,
        space_init: Vec::new(),
        ascii_init: [None; 128],
        added_matcher: None,
        split_bytes: [0; crate::bpe::sentencepiece::NUM_SPLIT_BYTES],
        split_safe: Vec::new(),
        cross_pieces: Vec::new(),
        cross_prev: [0; 4],
    };
    model.norm_added_tokens = norm_added_tokens
        .into_iter()
        .map(|mut spec| {
            spec.content = model.apply_norm_ops(&spec.content).into_owned();
            spec
        })
        .collect();
    model.finalize_speed_paths();
    Ok(model)
}

/// Translate a tokenizer.json `normalizer` into [`NormOp`]s, erroring on
/// anything unsupported — silently skipping a normalizer would produce token
/// IDs that diverge from HF.
fn parse_sp_normalizer(n: &NormalizerJson, out: &mut Vec<NormOp>) -> Result<()> {
    match n.kind.as_str() {
        "Sequence" => {
            for child in &n.normalizers {
                parse_sp_normalizer(child, out)?;
            }
        }
        "Prepend" => {
            let prefix = n
                .prepend
                .clone()
                .ok_or_else(|| eyre::eyre!("Prepend normalizer without a `prepend` string"))?;
            out.push(NormOp::Prepend(prefix));
        }
        "Replace" => {
            let content = n
                .content
                .clone()
                .ok_or_else(|| eyre::eyre!("Replace normalizer without a `content` string"))?;
            match &n.pattern {
                Some(PatternJson {
                    literal: Some(pattern),
                    ..
                }) => out.push(NormOp::Replace {
                    pattern: pattern.clone(),
                    content,
                }),
                // transformers' SpmConverter emits this exact regex for
                // sentencepiece's `remove_extra_whitespaces`.
                Some(PatternJson {
                    regex: Some(re), ..
                }) if re == " {2,}" => out.push(NormOp::CollapseSpaces { content }),
                Some(PatternJson {
                    regex: Some(re), ..
                }) => {
                    return Err(eyre::eyre!(
                        "Unsupported Replace normalizer regex: {re:?} (only \" {{2,}}\" is supported)"
                    ));
                }
                _ => return Err(eyre::eyre!("Replace normalizer without a pattern")),
            }
        }
        "Strip" => out.push(NormOp::Strip {
            left: n.strip_left.unwrap_or(true),
            right: n.strip_right.unwrap_or(true),
        }),
        "Precompiled" => {
            use base64::Engine;
            let b64 = n.precompiled_charsmap.as_deref().ok_or_else(|| {
                eyre::eyre!("Precompiled normalizer without a `precompiled_charsmap`")
            })?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .context("Failed to base64-decode precompiled_charsmap")?;
            let precompiled = spm_precompiled::Precompiled::from(&bytes)
                .map_err(|e| eyre::eyre!("Failed to parse precompiled_charsmap: {e}"))?;
            out.push(NormOp::Precompiled(
                crate::bpe::sentencepiece::PrecompiledCharsmap::new(precompiled),
            ));
        }
        other => {
            return Err(eyre::eyre!(
                "Unsupported normalizer type for SentencePiece tokenizers: {other}"
            ));
        }
    }
    Ok(())
}

/// Translate a tokenizer.json `pre_tokenizer` into a [`Metaspace`] config.
/// `None` (no pre-tokenizer, e.g. Llama 2) leaves spaces to the normalizer
/// and lets merges cross word boundaries.
///
/// `norm_ops`: the already-parsed normalizer ops, used to prove that a
/// `Split` on a literal space is a no-op (gemma-3/4).
fn parse_sp_metaspace(
    pre_tokenizer: &Option<PreTokenizerJson>,
    norm_ops: &[NormOp],
) -> Result<Option<Metaspace>> {
    fn from_metaspace(pt: &PreTokenizerJson) -> Result<Metaspace> {
        ensure!(
            pt.replacement.as_deref().unwrap_or("\u{2581}") == "\u{2581}",
            "Unsupported Metaspace replacement: {:?} (expected \"▁\")",
            pt.replacement
        );
        let prepend = match (&pt.prepend_scheme, pt.add_prefix_space) {
            (Some(scheme), _) => match scheme.as_str() {
                "never" => PrependScheme::Never,
                "always" => PrependScheme::Always,
                "first" => PrependScheme::First,
                other => {
                    return Err(eyre::eyre!("Unsupported Metaspace prepend_scheme: {other}"));
                }
            },
            (None, Some(false)) => PrependScheme::Never,
            (None, _) => PrependScheme::Always,
        };
        Ok(Metaspace {
            prepend,
            split: pt.split.unwrap_or(true),
        })
    }

    let Some(pt) = pre_tokenizer else {
        return Ok(None);
    };
    match pt.kind.as_str() {
        "Metaspace" => Ok(Some(from_metaspace(pt)?)),
        "Sequence"
            if pt.pretokenizers.len() == 1 && pt.pretokenizers[0].kind == "Metaspace" =>
        {
            Ok(Some(from_metaspace(&pt.pretokenizers[0])?))
        }
        // gemma-3/4: `Split` on a literal " " with MergedWithPrevious. The
        // normalizer has already replaced every space with "\u{2581}", so
        // the Split never matches and the model BPE-merges across word
        // boundaries exactly as with no pre-tokenizer at all. Accept it
        // only when a norm op proves all spaces are gone by then.
        "Split"
            if matches!(
                &pt.pattern,
                Some(PatternJson { literal: Some(l), .. }) if l == " "
            ) && pt.behavior.as_deref() == Some("MergedWithPrevious")
                && norm_ops.iter().any(|op| matches!(
                    op,
                    NormOp::Replace { pattern, content }
                        if pattern == " " && !content.contains(' ')
                )) =>
        {
            Ok(None)
        }
        other => Err(eyre::eyre!(
            "Unsupported pre_tokenizer type for SentencePiece tokenizers: {other}"
        )),
    }
}

// ---------------------------------------------------------------------------
// Normalizer detection
// ---------------------------------------------------------------------------

/// Determine whether the tokenizer's normalizer is NFC (the only kind we
/// support for ByteLevel BPE). Returns `true` for NFC, `false` for no
/// normalizer, and an error for anything else — silently skipping an unknown
/// normalizer would produce token IDs that diverge from HF.
fn detect_nfc_normalizer(normalizer: &Option<NormalizerJson>) -> Result<bool> {
    fn is_nfc(n: &NormalizerJson) -> Result<bool> {
        match n.kind.as_str() {
            "NFC" => Ok(true),
            "Sequence" => n
                .normalizers
                .iter()
                .try_fold(false, |acc, c| Ok(acc | is_nfc(c)?)),
            other => Err(eyre::eyre!("Unsupported normalizer type: {other}")),
        }
    }
    normalizer.as_ref().map_or(Ok(false), is_nfc)
}

// ---------------------------------------------------------------------------
// Pre-tokenizer detection
// ---------------------------------------------------------------------------

/// Determine the pretokenization scheme from a tokenizer.json `pre_tokenizer`.
///
/// Handles a bare `ByteLevel` (GPT-2 style, `use_regex: true`) and
/// `Sequence`s whose `Split` regexes (in order) form a known scheme —
/// either a single known regex or DeepSeek's digits/CJK/main triple.
fn detect_pretokenizer_type(
    pre_tokenizer: &Option<PreTokenizerJson>,
) -> Result<crate::pretokenize::PretokenizerType> {
    use crate::pretokenize::PretokenizerType;

    fn collect_split_regexes<'a>(pt: &'a PreTokenizerJson, out: &mut Vec<&'a str>) {
        if pt.kind == "Split"
            && let Some(PatternJson { regex: Some(re), .. }) = &pt.pattern
        {
            out.push(re);
        }
        for child in &pt.pretokenizers {
            collect_split_regexes(child, out);
        }
    }

    let Some(pt) = pre_tokenizer else {
        // No pre_tokenizer at all; keep the historical default.
        return Ok(PretokenizerType::GPT2);
    };
    // `BertPreTokenizer` carries no regex at all — it is two hardcoded splits —
    // so it is identified by kind name, not through `from_split_regexes`.
    fn has_bert(pt: &PreTokenizerJson) -> bool {
        pt.kind == "BertPreTokenizer" || pt.pretokenizers.iter().any(has_bert)
    }
    if has_bert(pt) {
        return Ok(PretokenizerType::Bert);
    }
    let mut regexes = Vec::new();
    collect_split_regexes(pt, &mut regexes);
    if regexes.is_empty() {
        // A bare ByteLevel. `use_regex: true` (the default) splits with the
        // GPT-2 regex; `use_regex: false` does no splitting at all — the
        // whitespace-lifted "superword" scheme a SuperBPE tokenizer needs so
        // its learned cross-whitespace merges fire.
        if pt.kind == "ByteLevel" {
            return Ok(if pt.use_regex == Some(false) {
                PretokenizerType::Superword
            } else {
                PretokenizerType::GPT2
            });
        }
        return Err(eyre::eyre!(
            "Unsupported pre_tokenizer type: {} (no Split regex found)",
            pt.kind
        ));
    }
    PretokenizerType::from_split_regexes(&regexes).ok_or_else(|| {
        eyre::eyre!("Unknown pre_tokenizer Split regexes, no fast pretokenizer for: {regexes:?}")
    })
}

/// Whether a `ByteLevel` pre-tokenizer anywhere in the chain sets
/// `add_prefix_space` (see the `Tokenizer::add_prefix_space` field for the
/// semantics).
fn detect_add_prefix_space(pre_tokenizer: &Option<PreTokenizerJson>) -> bool {
    fn walk(pt: &PreTokenizerJson) -> bool {
        (pt.kind == "ByteLevel" && pt.add_prefix_space == Some(true))
            || pt.pretokenizers.iter().any(walk)
    }
    pre_tokenizer.as_ref().is_some_and(walk)
}

// ---------------------------------------------------------------------------
// GPT-2 / ByteLevel BPE loader
// ---------------------------------------------------------------------------

/// Build the GPT-2 byte-to-unicode mapping table.
/// Returns (byte_to_unicode, unicode_to_byte).
fn build_byte_unicode_tables() -> ([char; 256], HashMap<char, u8>) {
    let allowed: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut b2u = ['\0'; 256];
    for &b in &allowed {
        b2u[b as usize] = b as char;
    }
    let mut n = 0u32;
    for b in 0..=255u8 {
        if b2u[b as usize] == '\0' {
            b2u[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    let u2b: HashMap<char, u8> = b2u.iter().enumerate().map(|(i, &c)| (c, i as u8)).collect();
    (b2u, u2b)
}

/// Decode a GPT-2 ByteLevel unicode string back to raw bytes.
///
/// Byte-level vocab strings consist solely of table chars; a string with any
/// other char is stored raw (e.g. DeepSeek V4 keeps its special tokens
/// unencoded in `model.vocab`) and taken as literal UTF-8 content.
fn unicode_to_bytes(s: &str, u2b: &HashMap<char, u8>) -> Vec<u8> {
    if s.chars().all(|c| u2b.contains_key(&c)) {
        s.chars().map(|c| u2b[&c]).collect()
    } else {
        s.as_bytes().to_vec()
    }
}

/// The GPT-2 mergeable ranks: the raw bytes of every `model.vocab` entry that
/// is not an added token, in ID order.
///
/// That list is the content of OpenAI's `r50k_base.tiktoken` — GPT-2 and
/// r50k_base are the same encoding, and the HF export stores the same 50256
/// ranks in the same order, only escaped through the ByteLevel unicode table
/// (rank 0 is `!` on both sides). So the tiktoken rank-file tests can be driven
/// from the committed GPT-2 fixture (`test_hub::gpt2_tokenizer_json`) instead of
/// an optional `~/data/tokenizers/r50k_base.tiktoken` download, and run
/// everywhere rather than skipping.
///
/// The added tokens have to come out: a rank file holds mergeable ranks only,
/// and `Tokenizer::from_ranks` requires every multi-byte entry to decompose into
/// exactly two lower ranks, which `<|endoftext|>` cannot.
#[cfg(test)]
pub(crate) fn gpt2_mergeable_ranks() -> Result<Vec<Vec<u8>>> {
    let tj = read_tokenizer_json(crate::test_hub::gpt2_tokenizer_json())?;
    let (_b2u, u2b) = build_byte_unicode_tables();
    let added: std::collections::HashSet<u32> = tj.added_tokens.iter().map(|t| t.id).collect();
    let mut ranks: Vec<(u32, Vec<u8>)> = tj
        .model
        .vocab
        .iter()
        .filter(|(_, id)| !added.contains(id))
        .map(|(tok_str, &id)| (id, unicode_to_bytes(tok_str, &u2b)))
        .collect();
    ranks.sort_unstable_by_key(|&(id, _)| id);
    // A rank file's id column is its line index, so the surviving IDs must be
    // dense from 0. Checked rather than assumed: a vocab with a gap would
    // otherwise silently shift every rank past it.
    for (i, (id, _)) in ranks.iter().enumerate() {
        ensure!(
            *id == i as u32,
            "GPT-2 rank {id} at position {i}: mergeable ranks must be dense"
        );
    }
    Ok(ranks.into_iter().map(|(_, bytes)| bytes).collect())
}

/// Load a HuggingFace `tokenizer.json` that uses ByteLevel BPE without
/// byte_fallback (e.g. GPT-2, RoBERTa).
///
/// Returns a [`bpe::tiktoken::Tokenizer`] with byte remapping.
/// WordPiece files load through here too (they build the same `Tokenizer`), so
/// one entry point covers every non-byte_fallback `tokenizer.json`.
pub fn load_hf_bpe(path: impl AsRef<Path>) -> Result<bpe::tiktoken::Tokenizer> {
    let path = path.as_ref();
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    build_bpe_or_wordpiece(&data).with_context(|| format!("Failed to load {}", path.display()))
}

/// Dispatch on the declared model family, for the entry points that hand back a
/// [`bpe::tiktoken::Tokenizer`].
fn build_bpe_or_wordpiece(data: &[u8]) -> Result<bpe::tiktoken::Tokenizer> {
    let family = probe_model_family(data)?;
    let tj = parse_tokenizer_json(data)?;
    match family {
        ModelFamily::WordPiece => build_wordpiece(&tj),
        ModelFamily::Bpe => build_bpe(&tj),
    }
}

/// Build a WordPiece tokenizer (BERT and family).
///
/// Returns the same [`bpe::tiktoken::Tokenizer`] a BPE file does, with the
/// WordPiece model installed in place of the merge table — so it rides the
/// existing `HfTokenizer::Bpe` arm, `batch.rs`'s existing worker pool, and the
/// existing PyO3 class with no dispatch changes anywhere.
fn build_wordpiece(tj: &TokenizerJson) -> Result<bpe::tiktoken::Tokenizer> {
    use crate::bpe::wordpiece::{DEFAULT_MAX_INPUT_CHARS_PER_WORD, WordPiece};

    let unk_token = tj
        .model
        .unk_token
        .as_deref()
        .ok_or_else(|| eyre::eyre!("WordPiece model without an `unk_token`"))?;
    let prefix = tj.model.continuing_subword_prefix.as_deref().unwrap_or("##");
    let max_input_chars = tj
        .model
        .max_input_chars_per_word
        .unwrap_or(DEFAULT_MAX_INPUT_CHARS_PER_WORD);

    // WordPiece vocab strings are literal UTF-8 — no GPT-2 byte-level
    // remapping, which is also why `ByteRemapping::from_byte_vocab` must not be
    // called: it requires a single-byte token for every UTF-8-legal byte, and a
    // WordPiece vocab has no such entries.
    let max_id = tj.model.vocab.values().max().copied().unwrap_or(0) as usize;
    let mut vocab: Vec<Arc<[u8]>> = vec![Arc::from(Vec::new().as_slice()); max_id + 1];
    for (tok_str, &id) in &tj.model.vocab {
        vocab[id as usize] = tok_str.as_bytes().into();
    }
    extend_vocab_with_added_tokens(&mut vocab, &tj.added_tokens);

    let wordpiece = WordPiece::new(
        tj.model
            .vocab
            .iter()
            .map(|(piece, &id)| (piece.clone(), TokenId::from(id))),
        unk_token,
        prefix,
        max_input_chars,
    )?;

    let vocab: Vec<Vec<u8>> = vocab.into_iter().map(|a| a.to_vec()).collect();
    let mut tokenizer = bpe::tiktoken::Tokenizer::new_wordpiece(Arc::new(wordpiece), vocab);
    tokenizer.set_pretokenizer_type(detect_pretokenizer_type(&tj.pre_tokenizer)?);
    tokenizer.set_bert_normalizer(parse_bert_normalizer(&tj.normalizer)?.map(Arc::new));
    tokenizer.set_wordpiece_decoder(parse_wordpiece_decoder(&tj.decoder).map(Arc::new));
    tokenizer.set_added_tokens(
        tj.added_tokens
            .iter()
            .map(|t| bpe::tiktoken::AddedTokenDef {
                content: t.content.as_bytes().into(),
                id: TokenId::from(t.id),
                lstrip: t.lstrip,
                rstrip: t.rstrip,
            })
            .collect(),
    );
    Ok(tokenizer)
}

/// The `WordPiece` decoder config, or `None` when the file declares no decoder
/// (or one of another kind — a `Sequence` is searched for a `WordPiece` member).
///
/// `None` is meaningful rather than a fallback: HF decodes a WordPiece file with
/// no decoder by joining pieces with spaces and leaving the `##` markers in
/// place. Defaulting to "splice and clean up" was wrong for such a file, and for
/// any file that sets `cleanup: false`.
fn parse_wordpiece_decoder(
    decoder: &Option<DecoderJson>,
) -> Option<crate::bpe::wordpiece::WordPieceDecoder> {
    fn find(d: &DecoderJson) -> Option<&DecoderJson> {
        match d.kind.as_str() {
            "WordPiece" => Some(d),
            "Sequence" => d.decoders.iter().find_map(find),
            _ => None,
        }
    }
    let d = find(decoder.as_ref()?)?;
    Some(crate::bpe::wordpiece::WordPieceDecoder {
        // HF's own defaults for the two fields, applied only once we know a
        // WordPiece decoder is present.
        prefix: d.prefix.as_deref().unwrap_or("##").into(),
        cleanup: d.cleanup.unwrap_or(true),
    })
}

/// Translate a `BertNormalizer` (possibly inside a `Sequence`) into its config.
/// `None` when the file declares no normalizer at all.
///
/// An unrelated normalizer type is an error rather than a silent skip: BERT's
/// token stream depends on lowercasing and accent stripping, so ignoring a
/// normalizer we do not model would mis-encode quietly.
fn parse_bert_normalizer(
    normalizer: &Option<NormalizerJson>,
) -> Result<Option<crate::bpe::bert_normalizer::BertNormalizer>> {
    use crate::bpe::bert_normalizer::BertNormalizer;

    fn find<'a>(n: &'a NormalizerJson) -> Result<Option<&'a NormalizerJson>> {
        match n.kind.as_str() {
            "BertNormalizer" => Ok(Some(n)),
            "Sequence" => {
                for child in &n.normalizers {
                    if let Some(found) = find(child)? {
                        return Ok(Some(found));
                    }
                }
                Ok(None)
            }
            other => Err(eyre::eyre!(
                "Unsupported normalizer type for WordPiece tokenizers: {other}"
            )),
        }
    }

    let Some(n) = normalizer else {
        return Ok(None);
    };
    let Some(bn) = find(n)? else {
        return Ok(None);
    };
    let norm = BertNormalizer::new(
        bn.clean_text.unwrap_or(true),
        bn.handle_chinese_chars.unwrap_or(true),
        // `null` is not "false": HF resolves it to the `lowercase` flag, which
        // is how bert-base-uncased strips accents without saying so.
        bn.strip_accents,
        bn.lowercase.unwrap_or(true),
    );
    Ok((!norm.is_noop()).then_some(norm))
}

fn build_bpe(tj: &TokenizerJson) -> Result<bpe::tiktoken::Tokenizer> {
    ensure!(
        tj.model.model_type == "BPE",
        "Unsupported model type: {} (expected BPE)",
        tj.model.model_type
    );
    ensure!(
        !tj.model.byte_fallback,
        "byte_fallback tokenizers should use load_hf_sentencepiece instead"
    );

    let (_b2u, u2b) = build_byte_unicode_tables();

    // Build vocab sorted by ID — each entry is the raw bytes for that token
    let max_id = tj.model.vocab.values().max().copied().unwrap_or(0) as usize;
    let mut vocab: Vec<Arc<[u8]>> = vec![Arc::from(Vec::new().as_slice()); max_id + 1];
    let mut vocab_inv: HashMap<Arc<[u8]>, TokenId, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(tj.model.vocab.len(), FxBuildHasher);
    for (tok_str, &id) in &tj.model.vocab {
        let bytes: Arc<[u8]> = unicode_to_bytes(tok_str, &u2b).into();
        vocab[id as usize] = bytes.clone();
        vocab_inv.insert(bytes, TokenId::from(id));
    }

    extend_vocab_with_added_tokens(&mut vocab, &tj.added_tokens);

    // Build merges from the merge list. Each merge "a b" means:
    // look up token IDs for "a" and "b", the merged token is vocab[concat(a,b)].
    let mut entries: Vec<(TokenId, TokenId, TokenId)> = Vec::with_capacity(tj.model.merges.len());
    for [str_a, str_b] in &tj.model.merges {
        let bytes_a = unicode_to_bytes(str_a, &u2b);
        let bytes_b = unicode_to_bytes(str_b, &u2b);
        let id_a = match vocab_inv.get(bytes_a.as_slice()) {
            Some(&id) => id,
            None => continue,
        };
        let id_b = match vocab_inv.get(bytes_b.as_slice()) {
            Some(&id) => id,
            None => continue,
        };
        let mut merged_bytes = bytes_a;
        merged_bytes.extend_from_slice(&bytes_b);
        let id_merged = match vocab_inv.get(merged_bytes.as_slice()) {
            Some(&id) => id,
            None => continue,
        };
        entries.push((id_a, id_b, id_merged));
    }

    let byte_remapping = bpe::ByteRemapping::from_byte_vocab(&vocab)?;
    let vocab: Vec<Vec<u8>> = vocab.into_iter().map(|a| a.to_vec()).collect();

    // The fast merge loops take the merged token's ID as the merge priority,
    // which is only correct when the merge list produces IDs in rank order
    // (true for every tiktoken-style vocab: GPT-2, cl100k, o200k, Qwen,
    // Llama-3, ...). Fairseq-heritage vocabs (RoBERTa/OPT/DeBERTa) order IDs
    // by corpus frequency instead; those carry their explicit list position
    // as the rank.
    let id_order_ok = entries.is_sorted_by_key(|&(_, _, merged)| merged);
    let mut tokenizer = if id_order_ok {
        let mut merges: HashMap<(TokenId, TokenId), TokenId, FxBuildHasher> =
            HashMap::with_capacity_and_hasher(entries.len(), FxBuildHasher);
        for (id_a, id_b, id_merged) in entries {
            merges.entry((id_a, id_b)).or_insert(id_merged);
        }
        bpe::tiktoken::Tokenizer::new(merges, vocab, byte_remapping)
    } else {
        let mut merges: bpe::tiktoken::RankedMerges =
            HashMap::with_capacity_and_hasher(entries.len(), FxBuildHasher);
        for (rank, (id_a, id_b, id_merged)) in entries.into_iter().enumerate() {
            merges
                .entry(bpe::ranked_merge_key(id_a, id_b))
                .or_insert((id_merged, rank as u32));
        }
        bpe::tiktoken::Tokenizer::new_ranked(merges, vocab, byte_remapping)
    };
    tokenizer.set_pretokenizer_type(detect_pretokenizer_type(&tj.pre_tokenizer)?);
    tokenizer.set_normalize_nfc(detect_nfc_normalizer(&tj.normalizer)?);
    tokenizer.set_add_prefix_space(detect_add_prefix_space(&tj.pre_tokenizer));
    tokenizer.set_ignore_merges(tj.model.ignore_merges);
    // All added tokens (special and non-special) are matched atomically in the
    // raw input by HF's AddedVocabulary; mirror that, including the
    // whitespace-stripping flags.
    tokenizer.set_added_tokens(
        tj.added_tokens
            .iter()
            .map(|t| bpe::tiktoken::AddedTokenDef {
                content: t.content.as_bytes().into(),
                id: TokenId::from(t.id),
                lstrip: t.lstrip,
                rstrip: t.rstrip,
            })
            .collect(),
    );
    // Last, by contract: it reads the scheme and `ignore_merges` set above.
    // A no-op unless this is a SuperBPE tokenizer, and never changes the
    // token stream — see `bpe::superword`.
    tokenizer.enable_superword_two_level();
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_str_to_bytes() {
        assert_eq!(token_str_to_bytes("<0x00>"), vec![0x00]);
        assert_eq!(token_str_to_bytes("<0xFF>"), vec![0xFF]);
        assert_eq!(token_str_to_bytes("<0x0A>"), vec![0x0A]);
        assert_eq!(token_str_to_bytes("hello"), b"hello".to_vec());
        assert_eq!(token_str_to_bytes("▁the"), "▁the".as_bytes().to_vec());
        assert_eq!(token_str_to_bytes("▁"), "▁".as_bytes().to_vec());
        assert_eq!(token_str_to_bytes("<unk>"), b"<unk>".to_vec());
        assert_eq!(token_str_to_bytes("<s>"), b"<s>".to_vec());
    }

    #[test]
    fn test_parse_legacy_model_without_type() {
        // Pre-tokenizers-0.9 files have no `model.type`; they must parse as BPE.
        let json = br#"{"model": {"vocab": {"a": 0}, "merges": []}}"#;
        let tj = parse_tokenizer_json(json).unwrap();
        assert_eq!(tj.model.model_type, "BPE");
    }

    /// WordPiece must be recognised both when typed and — as in
    /// `bert-base-uncased`, whose `model` block has **no** `type` field — from
    /// the `max_input_chars_per_word` marker alone. `continuing_subword_prefix`
    /// must NOT be the discriminator: BPE serializes it too.
    #[test]
    fn test_wordpiece_family_is_detected() {
        // Three hashes: the JSON contains `"##`, which would close both a
        // `br#"` and a `br##"` raw string.
        let typed = br###"{"model": {"type": "WordPiece", "unk_token": "[UNK]",
            "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
            "vocab": {"[UNK]": 0, "hello": 1}}}"###;
        let untyped: &[u8] = b"{\"model\": {\"unk_token\": \"[UNK]\",
            \"continuing_subword_prefix\": \"##\", \"max_input_chars_per_word\": 100,
            \"vocab\": {\"[UNK]\": 0}}}";
        for json in [&typed[..], untyped] {
            assert_eq!(probe_model_family(json).unwrap(), ModelFamily::WordPiece);
        }
        // A BPE file that happens to carry `continuing_subword_prefix` (the
        // original gpt2 upload does) stays BPE.
        let gpt2ish = br#"{"model": {"vocab": {"a": 0}, "merges": [],
            "continuing_subword_prefix": ""}}"#;
        assert_eq!(probe_model_family(gpt2ish).unwrap(), ModelFamily::Bpe);
    }

    /// A minimal WordPiece file must load and encode through the whole
    /// pipeline: BertNormalizer → bert pretokenizer → MaxMatch.
    #[test]
    fn test_build_wordpiece_end_to_end() {
        let json = br###"{
            "normalizer": {"type": "BertNormalizer", "clean_text": true,
                "handle_chinese_chars": true, "strip_accents": null, "lowercase": true},
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
                "vocab": {"[UNK]": 0, "un": 1, "##aff": 2, "##able": 3, "cafe": 4,
                          "!": 5, "hello": 6}}
        }"###;
        let mut tok = build_bpe_or_wordpiece(json).expect("WordPiece must load");
        assert_eq!(tok.pretokenizer_type(), crate::pretokenize::PretokenizerType::Bert);
        assert!(tok.wordpiece().is_some());

        let mut ids = Vec::new();
        // "Unaffable" lowercases; "Café" strips its accent; "!" isolates;
        // "zzz" is unknown and collapses to one [UNK].
        tok.encode_with_added_tokens_flat("Unaffable Café! zzz".as_bytes(), &mut ids);
        assert_eq!(ids, vec![1, 2, 3, 4, 5, 0]);

        // Decode is HF's, not a byte concatenation — and this file declares no
        // `decoder`, so HF joins with spaces and keeps the `##` markers. The
        // splice and the cleanup are the decoder's doing, not defaults.
        let tokens: Vec<TokenId> = ids.iter().map(|&i| TokenId::from(i)).collect();
        assert_eq!(
            tok.decode_wordpiece(&tokens).unwrap(),
            "un ##aff ##able cafe ! [UNK]"
        );
    }

    /// The `decoder` block drives both the ` ##` splice and the `cleanup` tidy;
    /// each combination decodes differently. Every expectation measured against
    /// `tokenizers` 0.22.2 (see `WordPieceDecoder`'s table).
    #[test]
    fn test_wordpiece_decoder_block_is_honored() {
        let with_decoder = |decoder: &str| {
            let json = format!(
                r###"{{
                    "pre_tokenizer": {{"type": "BertPreTokenizer"}},
                    {decoder}
                    "model": {{"type": "WordPiece", "unk_token": "[UNK]",
                        "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
                        "vocab": {{"[UNK]": 0, "ab": 1, "##cd": 2, "hello": 3, ".": 4}}}}
                }}"###
            );
            let tok = build_bpe_or_wordpiece(json.as_bytes()).expect("must load");
            let ids: Vec<TokenId> = [1u32, 2, 3, 4].iter().map(|&i| TokenId::from(i)).collect();
            tok.decode_wordpiece(&ids).unwrap()
        };

        assert_eq!(with_decoder(""), "ab ##cd hello .", "no decoder: space join only");
        assert_eq!(
            with_decoder(r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},"###),
            "abcd hello."
        );
        assert_eq!(
            with_decoder(r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false},"###),
            "abcd hello ."
        );
        // HF's own defaults when the fields are omitted from a present block.
        assert_eq!(
            with_decoder(r#""decoder": {"type": "WordPiece"},"#),
            "abcd hello."
        );
        // A decoder of another kind is not a WordPiece decoder.
        assert_eq!(
            with_decoder(r#""decoder": {"type": "ByteLevel"},"#),
            "ab ##cd hello ."
        );
        // ... but one nested in a Sequence is found.
        assert_eq!(
            with_decoder(
                r###""decoder": {"type": "Sequence", "decoders": [{"type": "ByteLevel"},
                    {"type": "WordPiece", "prefix": "##", "cleanup": false}]},"###
            ),
            "abcd hello ."
        );
    }

    /// A WordPiece vocab has no single-byte tokens, so the byte-level remapping
    /// must never be attempted — and the cache seed must agree with the miss
    /// path, which is what a wrong seed would break silently.
    #[test]
    fn test_wordpiece_seed_matches_miss_path() {
        let json = br###"{
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
                "vocab": {"[UNK]": 0, "ab": 1, "##cd": 2, "abcd": 3, "x": 4}}
        }"###;
        let mut tok = build_bpe_or_wordpiece(json).unwrap();
        // "abcd" is a whole vocab word AND decomposable as ab+##cd. MaxMatch is
        // longest-first, so it must be the single ID 3 — and that is exactly
        // what the vocab seed put in the cache, so the seeded hit and a cold
        // miss cannot disagree.
        let mut ids = Vec::new();
        tok.encode_with_added_tokens_flat(b"abcd", &mut ids);
        assert_eq!(ids, vec![3]);

        let mut fresh = build_bpe_or_wordpiece(json).unwrap();
        let mut ids2 = Vec::new();
        // A pretoken absent from the vocab takes the true miss path.
        fresh.encode_with_added_tokens_flat(b"abcdx", &mut ids2);
        assert_eq!(ids2, vec![0], "abcdx has no ##x, so the whole word is UNK");
    }

    /// Unsupported model families must be refused by name, not with the
    /// deserializer's shape error for the BPE schema (Unigram's vocab is a
    /// `[piece, score]` list, not a map).
    ///
    /// WordPiece used to be in this list and is now supported, so the rows for
    /// it moved to [`test_wordpiece_family_is_detected`].
    #[test]
    fn test_unsupported_model_type_named_in_error() {
        let unigram = br#"{"model": {"type": "Unigram", "unk_id": 0,
            "vocab": [["<unk>", 0.0], ["hello", -3.1]]}}"#;
        // Pre-0.9 tokenizers files omit model.type; the family is inferred
        // from its marker fields (the t5-small shape).
        let untyped_unigram = br#"{"model": {"unk_id": 0, "vocab": [["<unk>", 0.0]]}}"#;
        for (json, name) in [
            (&unigram[..], "Unigram"),
            (&untyped_unigram[..], "Unigram (untyped legacy file)"),
        ] {
            let err = match parse_tokenizer_json(json) {
                Ok(_) => panic!("expected {name} to be refused"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains(&format!("Unsupported model type \"{name}\"")),
                "unhelpful {name} error: {err}"
            );
            assert!(!err.contains("missing field"), "shape error leaked: {err}");
        }
    }

    #[test]
    fn test_parse_error_names_the_field() {
        // The first line of the error must carry the deserializer's detail,
        // not just a generic "failed to parse".
        let json = br#"{"model": {"type": "BPE", "merges": []}}"#;
        let err = match parse_tokenizer_json(json) {
            Ok(_) => panic!("expected a parse error"),
            Err(e) => e,
        };
        let first_line = err.to_string();
        assert!(first_line.contains("vocab"), "unhelpful error: {first_line}");
    }

    fn tinyllama_path() -> Option<std::path::PathBuf> {
        let path = crate::test_hub::hf_tokenizer_json("TinyLlama/TinyLlama-1.1B-Chat-v1.0");
        if path.is_none() {
            eprintln!("Skipping: TinyLlama tokenizer.json not in the HF cache");
        }
        path
    }

    #[test]
    fn test_load_tinyllama_sentencepiece() {
        let Some(path) = tinyllama_path() else { return };
        let tokenizer = load_hf_sentencepiece(path).unwrap();
        eprintln!("{:?}", tokenizer);
    }

    #[test]
    fn test_encode_hello_sentencepiece() {
        let Some(path) = tinyllama_path() else { return };
        let tokenizer = load_hf_sentencepiece(path).unwrap();
        let ids = tokenizer.encode_raw("Hello world");
        eprintln!("Encoded: {:?}", ids);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, b"Hello world");
    }

    /// Build a minimal ByteLevel tokenizer.json: the 256 byte tokens (ID ==
    /// byte value), plus `extra_vocab` entries and `merges` given as raw
    /// text (converted to the GPT-2 unicode encoding here), plus raw
    /// `added_tokens` JSON.
    fn byte_level_json(
        extra_vocab: &[(&str, u32)],
        merges: &[(&str, &str)],
        added_tokens_json: &str,
    ) -> Vec<u8> {
        byte_level_json_with_pretok(extra_vocab, merges, added_tokens_json, r#"{"type": "ByteLevel"}"#)
    }

    fn byte_level_json_with_pretok(
        extra_vocab: &[(&str, u32)],
        merges: &[(&str, &str)],
        added_tokens_json: &str,
        pre_tokenizer_json: &str,
    ) -> Vec<u8> {
        let (b2u, _) = build_byte_unicode_tables();
        let esc = |s: String| -> String { s.replace('\\', "\\\\").replace('"', "\\\"") };
        let enc = |s: &str| -> String { esc(s.bytes().map(|b| b2u[b as usize]).collect()) };
        let mut vocab_entries: Vec<String> = (0u16..=255)
            .map(|b| format!("\"{}\": {}", esc(b2u[b as usize].to_string()), b))
            .collect();
        for (text, id) in extra_vocab {
            vocab_entries.push(format!("\"{}\": {}", enc(text), id));
        }
        let merges_entries: Vec<String> = merges
            .iter()
            .map(|(a, b)| format!("[\"{}\", \"{}\"]", enc(a), enc(b)))
            .collect();
        format!(
            "{{\"added_tokens\": [{}], \"pre_tokenizer\": {}, \
             \"model\": {{\"type\": \"BPE\", \"vocab\": {{{}}}, \"merges\": [{}]}}}}",
            added_tokens_json,
            pre_tokenizer_json,
            vocab_entries.join(", "),
            merges_entries.join(", ")
        )
        .into_bytes()
    }

    fn encode_bpe(tok: &mut bpe::tiktoken::Tokenizer, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        tok.encode_with_added_tokens_flat(text.as_bytes(), &mut out);
        out
    }

    /// Merge priority must follow the merge list's order even when the
    /// merged token IDs do not (fairseq-heritage vocabs: RoBERTa/OPT).
    #[test]
    fn test_rank_mapped_merges_follow_list_order() {
        // Rank 0 produces ID 350, rank 1 produces ID 300: IDs are NOT in
        // rank order, so id-as-rank would apply "a"+"b" (300) before
        // "b"+"c" (350) and produce [300, 99] on "abc".
        let json = byte_level_json(
            &[("bc", 350), ("ab", 300)],
            &[("b", "c"), ("a", "b")],
            "",
        );
        let HfTokenizer::Bpe(mut tok) = load_hf_slice(&json).unwrap() else {
            panic!("expected ByteLevel BPE");
        };
        assert_eq!(encode_bpe(&mut tok, "abc"), vec![97, 350]);
        // Rank order also decides between two live candidates mid-word.
        assert_eq!(encode_bpe(&mut tok, "ab"), vec![300]);
        assert_eq!(encode_bpe(&mut tok, "bc"), vec![350]);

        // Same merges with IDs in rank order stay on the id-as-rank fast
        // path and agree.
        let json = byte_level_json(
            &[("bc", 300), ("ab", 350)],
            &[("b", "c"), ("a", "b")],
            "",
        );
        let HfTokenizer::Bpe(mut tok) = load_hf_slice(&json).unwrap() else {
            panic!("expected ByteLevel BPE");
        };
        assert_eq!(encode_bpe(&mut tok, "abc"), vec![97, 300]);
    }

    /// `lstrip`/`rstrip` added-token flags absorb the whitespace adjacent
    /// to a match, like HF's AddedVocabulary.
    #[test]
    fn test_added_token_lstrip_rstrip() {
        let added = r#"{"id": 400, "content": "<m>", "lstrip": true, "special": true},
                     {"id": 401, "content": "<r>", "rstrip": true, "special": true}"#;
        let json = byte_level_json(&[], &[], added);
        let HfTokenizer::Bpe(mut tok) = load_hf_slice(&json).unwrap() else {
            panic!("expected ByteLevel BPE");
        };
        // lstrip: the whitespace before the match is absorbed, including
        // multi-char and non-ASCII whitespace; whitespace after it is not.
        assert_eq!(encode_bpe(&mut tok, "a <m> b"), vec![97, 400, 32, 98]);
        assert_eq!(encode_bpe(&mut tok, "a \t\n<m>"), vec![97, 400]);
        assert_eq!(
            encode_bpe(&mut tok, "a\u{a0}<m>"),
            vec![97, 400],
            "U+00A0 is `\\s` whitespace and must be absorbed"
        );
        // rstrip: the whitespace after the match is absorbed.
        assert_eq!(encode_bpe(&mut tok, "a <r> b"), vec![97, 32, 401, 98]);
        assert_eq!(encode_bpe(&mut tok, "<r>\n\n\nb"), vec![401, 98]);
        // Back-to-back: <r>'s rstrip consumes the gap before <m>.
        assert_eq!(encode_bpe(&mut tok, "<r> <m>"), vec![401, 400]);
        // No flags on plain text.
        assert_eq!(encode_bpe(&mut tok, "a b"), vec![97, 32, 98]);
    }

    /// `ByteLevel(add_prefix_space=true)` (RoBERTa-style exports): every
    /// non-empty added-token-split segment gets a leading space; empty
    /// segments (adjacent added tokens, leading added token) do not.
    #[test]
    fn test_byte_level_add_prefix_space() {
        let added = r#"{"id": 400, "content": "<m>", "lstrip": true, "special": true}"#;
        let json = byte_level_json_with_pretok(
            &[],
            &[],
            added,
            r#"{"type": "ByteLevel", "add_prefix_space": true}"#,
        );
        let HfTokenizer::Bpe(mut tok) = load_hf_slice(&json).unwrap() else {
            panic!("expected ByteLevel BPE");
        };
        // "ab" -> " ab" (one pretoken), already-spaced input unchanged.
        assert_eq!(encode_bpe(&mut tok, "ab"), vec![32, 97, 98]);
        assert_eq!(encode_bpe(&mut tok, " ab"), vec![32, 97, 98]);
        // Each segment around an added token gets its own prefix; the empty
        // segment between adjacent tokens gets none (HF parity, verified
        // against tokenizers on obi/deid_roberta_i2b2).
        assert_eq!(encode_bpe(&mut tok, "a<m>b"), vec![32, 97, 400, 32, 98]);
        assert_eq!(encode_bpe(&mut tok, "<m><m>"), vec![400, 400]);
        // lstrip trim happens before the prefix is applied.
        assert_eq!(encode_bpe(&mut tok, "x <m> y"), vec![32, 120, 400, 32, 121]);
    }

    #[test]
    fn test_load_gpt2_from_hf() {
        let path = crate::test_hub::gpt2_tokenizer_json();
        let mut tokenizer = load_hf_bpe(&path).unwrap();
        eprintln!("{:?}", tokenizer);

        // Encode and verify roundtrip
        let text = b"Hello, world! This is a test.";
        let pretokens = crate::pretokenize::pretokenize_as_iter(text);
        let mut token_ids: Vec<TokenId> = Vec::new();
        tokenizer.memoized_encode(pretokens, |tokens| {
            token_ids.extend_from_slice(tokens);
        });
        eprintln!("Encoded {} bytes -> {:?}", text.len(), token_ids);
        let decoded: Vec<u8> = tokenizer.decode(&token_ids).collect();
        assert_eq!(decoded, text);
    }
}
