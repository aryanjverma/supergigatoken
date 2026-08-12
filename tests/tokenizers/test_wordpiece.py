"""WordPiece/BERT specifics that the shared parity suite cannot express.

The shared suite (test_hf_parity.py) already covers encode parity, added
tokens, and batch==single for every BERT spec. This file adds what is unique to
the family: the whole-word `[UNK]` rule, the char-counted length cap, the
`clean_text` classification table, the non-invertible decoder, and — the
strongest gate here — a sweep of *every* Unicode scalar through both
pretokenizers, which is what catches the places where HF's bundled UCD is
older than ICU's.
"""

import pytest
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordPiece

from gigatoken.gigatoken_rs import BPETokenizer


@pytest.fixture(scope="session")
def uncased(bert_base_uncased_tokenizer_path):
    return (
        Tokenizer.from_file(str(bert_base_uncased_tokenizer_path)),
        BPETokenizer.from_hf(bert_base_uncased_tokenizer_path),
    )


def _ids(hf, gt, text: str) -> tuple[list[int], list[int]]:
    return (
        hf.encode(text, add_special_tokens=False).ids,
        gt.encode(text.encode("utf-8")).tolist(),
    )


# ---------------------------------------------------------------------------
# Full-coverage pretokenizer sweep
# ---------------------------------------------------------------------------


def test_bert_matches_hf_for_every_codepoint(uncased):
    """Encode `"a<c>a"` for every Unicode scalar and require identical IDs.

    This is the gate on the hardcoded staleness deltas in
    `pretokenize::unicode`. HF's `is_bert_punc` and control predicates come from
    a crate whose UCD is several versions behind the `icu` crate this repo
    builds its tables from: 141 codepoints are `\\p{P}` to ICU but not to HF,
    and 2 go the other way. Getting one wrong moves a pretoken boundary, which
    changes the token stream — so the deltas are asserted against the installed
    `tokenizers`, not against another copy of Unicode.

    Both sides are batched (`encode_batch_fast` / `encode_batch`), so the whole
    1.1M-codepoint sweep is a few thousand calls rather than millions.
    """
    hf, gt = uncased
    scalars = [cp for cp in range(0x110000) if not (0xD800 <= cp <= 0xDFFF)]
    mismatches = []
    CHUNK = 8192
    for base in range(0, len(scalars), CHUNK):
        batch = scalars[base : base + CHUNK]
        probes = ["a" + chr(cp) + "a" for cp in batch]
        hf_ids = [e.ids for e in hf.encode_batch_fast(probes, add_special_tokens=False)]
        gt_ids = [a.tolist() for a in gt.encode_batch([p.encode("utf-8") for p in probes])]
        for cp, want, got in zip(batch, hf_ids, gt_ids):
            if want != got:
                mismatches.append((f"U+{cp:04X}", want, got))
    assert not mismatches, f"{len(mismatches)} codepoints diverged from HF: {mismatches[:20]}"


@pytest.mark.parametrize(
    "cp",
    [
        0x0378,  # unassigned: clean_text KEEPS it (ICU's "Other" group would not)
        0x0085,  # both whitespace and control: removed, not spaced
        0x000B,
        0x000C,
        0x200B,  # ZWSP: removed, joining the surrounding word
        0xFFFD,  # removed by value, not by category
        0x00A0,  # separator: becomes a space
        0x2028,
        0x09FD,  # \p{P} to ICU, NOT punctuation to HF (stale UCD)
        0x2E4F,
        0x166D,  # So, yet punctuation to HF
        0x111C9,  # Mn, yet punctuation to HF
        0x0890,  # Cf to ICU, kept by HF
        0x13430,
        0x00D7,  # non-ASCII symbols are word chars
        0x20AC,
        0x00A1,  # non-ASCII \p{P} is punctuation
        0x3001,
        0x0301,  # combining mark: word char, dropped by strip_accents
    ],
    ids=lambda cp: f"U+{cp:04X}",
)
def test_clean_text_and_punct_pins(uncased, cp):
    """The counter-intuitive codepoints, end to end through both pipelines."""
    hf, gt = uncased
    hf_ids, gt_ids = _ids(hf, gt, f"a{chr(cp)}b")
    assert gt_ids == hf_ids, f"diverged on U+{cp:04X}"


# ---------------------------------------------------------------------------
# WordPiece model semantics
# ---------------------------------------------------------------------------


def test_whole_word_unk(uncased):
    """A word with any unsegmentable part becomes ONE [UNK], not a partial
    emission — even when a long prefix of it is in the vocab."""
    hf, gt = uncased
    # U+0378 is unassigned, so clean_text keeps it and no vocab piece covers it.
    # Built with chr(): an unassigned codepoint written literally is easy to
    # mangle, and this is the case the test turns on.
    u = chr(0x378)
    for text in [f"zzzz{u}zzz", f"x{u}y", f"un{u}affable"]:
        hf_ids, gt_ids = _ids(hf, gt, text)
        assert gt_ids == hf_ids, text
        assert hf_ids.count(100) >= 1, f"expected an [UNK] in {text!r} -> {hf_ids}"


def _from_hf_object(hf: Tokenizer) -> BPETokenizer:
    """Load a hand-built `tokenizers.Tokenizer` into gigatoken through its
    serialized tokenizer.json, so both sides encode the same model."""
    from gigatoken.gigatoken_rs import load_hf_json

    tok = load_hf_json(hf.to_str())
    # Not just a type narrowing: a WordPiece file must ride the byte-level
    # backend, never the SentencePiece one.
    assert isinstance(tok, BPETokenizer), type(tok)
    return tok


@pytest.mark.parametrize("cap", [3, 4, 100])
def test_max_input_chars_per_word_counts_chars_not_bytes(cap):
    """The cap counts chars, not bytes: `cap` 2-byte chars is 2*cap bytes and
    must still segment, while cap+1 chars collapses to a single [UNK].

    Both sides are checked — the point is that gigatoken counts the same way HF
    does, which a byte-counting implementation would get wrong for any
    non-ASCII word.
    """
    vocab = {"[UNK]": 0, "é": 1, "##é": 2}
    hf = Tokenizer(WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=cap))
    hf.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    gt = _from_hf_object(hf)

    at_cap = "é" * cap
    over_cap = "é" * (cap + 1)
    assert hf.encode(at_cap, add_special_tokens=False).ids == [1] + [2] * (cap - 1)
    assert hf.encode(over_cap, add_special_tokens=False).ids == [0]
    assert gt.encode(at_cap.encode("utf-8")).tolist() == [1] + [2] * (cap - 1)
    assert gt.encode(over_cap.encode("utf-8")).tolist() == [0]


def test_whole_word_unk_discards_matched_prefix():
    """Failure anywhere makes the whole word one [UNK] — the already-matched
    pieces are discarded, not emitted."""
    vocab = {"[UNK]": 0, "ab": 1, "##cd": 2}
    hf = Tokenizer(WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    hf.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    gt = _from_hf_object(hf)
    for text, want in [("abcd", [1, 2]), ("abce", [0]), ("abc", [0]), ("abcdab", [0])]:
        assert hf.encode(text, add_special_tokens=False).ids == want, text
        assert gt.encode(text.encode("utf-8")).tolist() == want, text


def test_cjk_becomes_individual_pretokens(bert_multilingual_tokenizer_path):
    """handle_chinese_chars pads each CJK char with spaces, so the whitespace
    split makes every ideograph its own pretoken."""
    hf = Tokenizer.from_file(str(bert_multilingual_tokenizer_path))
    gt = BPETokenizer.from_hf(bert_multilingual_tokenizer_path)
    for text in ["中文", "中文x", "a中b文c", "日本語テスト", "汉字漢字"]:
        hf_ids, gt_ids = _ids(hf, gt, text)
        assert gt_ids == hf_ids, text
    # Kana is NOT in BERT's CJK ranges, so テスト stays one word.
    assert len(hf.encode("テスト", add_special_tokens=False).ids) < 3


def test_accents_stripped_only_when_configured(bert_base_uncased_tokenizer_path, bert_base_cased_tokenizer_path):
    """`strip_accents: null` follows `lowercase`, so the uncased model strips
    and the cased one does not — the same JSON value, opposite behaviour."""
    for path in (bert_base_uncased_tokenizer_path, bert_base_cased_tokenizer_path):
        hf = Tokenizer.from_file(str(path))
        gt = BPETokenizer.from_hf(path)
        for text in ["Café", "naïve", "Ω", "İstanbul", "ﬁ"]:
            hf_ids, gt_ids = _ids(hf, gt, text)
            assert gt_ids == hf_ids, f"{path.parent.name} {text!r}"

    lower = Tokenizer.from_file(str(bert_base_uncased_tokenizer_path))
    cased = Tokenizer.from_file(str(bert_base_cased_tokenizer_path))
    assert lower.encode("Café", add_special_tokens=False).tokens == ["cafe"]
    assert cased.encode("Café", add_special_tokens=False).tokens != ["cafe"]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def test_decode_matches_hf_wordpiece_decoder(uncased):
    """gigatoken's decode must equal HF's WordPiece decoder, including the
    `cleanup` spacing rules — not a concatenation of vocab bytes.

    `skip_special_tokens=False` is the apples-to-apples comparison: gigatoken
    decodes every ID it is given, while HF's default drops specials.
    """
    hf, gt = uncased
    for text in [
        "unaffable",
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "don't stop, really!",
        "a b c",
        "Café naïve",
        f"x{chr(0x378)}y unknown",  # yields an [UNK], which must survive on both sides
    ]:
        ids = hf.encode(text, add_special_tokens=False).ids
        want = hf.decode(ids, skip_special_tokens=False)
        assert gt.decode(ids).decode("utf-8") == want, text


def test_decode_is_not_invertible_but_is_stable(uncased):
    """Encoding is lossy for WordPiece (case, accents, spacing). Pin that the
    loss is exactly HF's, rather than pretending a roundtrip holds."""
    hf, gt = uncased
    text = "The   Quick\tBrown  Fox!"
    ids = gt.encode(text.encode("utf-8")).tolist()
    decoded = gt.decode(ids).decode("utf-8")
    assert decoded != text
    assert decoded == hf.decode(ids, skip_special_tokens=False)
    # Re-encoding the decoded form is a fixed point.
    assert gt.encode(decoded.encode("utf-8")).tolist() == ids


# ---------------------------------------------------------------------------
# Parallel/serial agreement
# ---------------------------------------------------------------------------


def test_parallel_and_serial_agree(uncased):
    """`parallel=False` must be byte-identical to the parallel path, including
    when one document is large enough to be split across workers."""
    hf, gt = uncased
    docs = [
        b"The quick brown fox jumps over the lazy dog. " * 200,
        "Café naïve 中文 test unaffable zzzqqq don't.".encode() * 200,
        b"short",
        b"",
    ]
    par = gt.encode_batch(docs)
    ser = gt.encode_batch(docs, parallel=False)
    for a, b in zip(par, ser):
        assert a.tolist() == b.tolist()
    for doc, batched in zip(docs, par):
        assert gt.encode(doc).tolist() == batched.tolist()
