"""gigatoken vs HuggingFace parity for the SuperBPE (superword) encoder.

The whole point of Axis 2: gigatoken must fast-encode a SuperBPE tokenizer
identically to HF. A SuperBPE tokenizer is a byte-level BPE whose whitespace
pretokenization is *lifted* (``ByteLevel(use_regex=False)``) so its learned
"superword" merges — which bridge whitespace — can fire. gigatoken loads that
via the new ``PretokenizerType::Superword`` (a no-split scheme); HF loads the
same ``tokenizer.json`` directly. This test trains a small SuperBPE, saves it
once, and asserts the two engines agree token-for-token on a range of texts,
including inputs where superwords span several words.
"""

import numpy as np
import pytest

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

import gigatoken
from gigatoken import train_superbpe

from conftest import gpt2_bytes_to_unicode as bytes_to_unicode

# A corpus with strong multi-word collocations so stage 2 actually learns
# superwords (tokens that bridge whitespace) to exercise the lifted scheme.
_SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "in the United States of America",
    "on the other hand it was clear",
    "as well as the rest of the world",
    "according to the report of the year",
    "machine learning and artificial intelligence",
    "the end of the beginning of the story",
    "one two three four five six seven eight",
]
CORPUS = ("\n".join(_SENTENCES * 200)).encode("utf-8")
VOCAB_SIZE = 600
TRANSITION_POINT = 450


def _superbpe_tokenizer_json(tmp_path):
    vocab, merges = train_superbpe(CORPUS, VOCAB_SIZE, TRANSITION_POINT, [])
    hf_vocab = {bytes_to_unicode(v): k for k, v in vocab.items()}
    hf_merges = [(bytes_to_unicode(a), bytes_to_unicode(b)) for a, b in merges]
    tok = Tokenizer(models.BPE(vocab=hf_vocab, merges=hf_merges))
    # use_regex=False lifts whitespace pretokenization -> superwords fire; the
    # loader maps this to PretokenizerType::Superword on the gigatoken side.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    path = str(tmp_path / "superbpe.json")
    tok.save(path)
    has_super = any(
        b == 0x20 and i > 0 and t[i - 1] != 0x20
        for t in vocab.values()
        for i, b in enumerate(t)
    )
    assert has_super, "corpus/params learned no superwords; test would be vacuous"
    return path, tok


@pytest.fixture(scope="module")
def superbpe_pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("superbpe")
    path, hf = _superbpe_tokenizer_json(tmp)
    giga = gigatoken.Tokenizer(path)
    return hf, giga


PARITY_TEXTS = [
    "the quick brown fox",
    "in the United States of America",
    "on the other hand",
    "according to the report",
    "machine learning and artificial intelligence",
    "the the the the the",
    "one two three four five",
    "hello world",
    "",
    " ",
    "   leading and trailing   ",
    "a b c d e f g",
    "no-collocations-here xyzzy plugh",
    "1234567890",
    "the end of the beginning\nof the story",
    "mixed the United States 世界 case",
]


@pytest.mark.parametrize("text", PARITY_TEXTS, ids=lambda t: repr(t)[:40])
def test_superword_encode_matches_hf(superbpe_pair, text):
    hf, giga = superbpe_pair
    hf_ids = hf.encode(text, add_special_tokens=False).ids
    giga_ids = giga.encode(text.encode("utf-8")).tolist()
    assert giga_ids == hf_ids, (
        f"Mismatch for {text!r}:\n  HF:    {hf_ids}\n  giga:  {giga_ids}"
    )


def test_superword_encode_batch_matches_hf(superbpe_pair):
    hf, giga = superbpe_pair
    docs = [t for t in PARITY_TEXTS if t]
    hf_batch = [e.ids for e in hf.encode_batch_fast(docs, add_special_tokens=False)]
    giga_batch = giga.encode_batch([d.encode("utf-8") for d in docs])
    for text, hf_ids, giga_arr in zip(docs, hf_batch, giga_batch):
        assert np.asarray(giga_arr).tolist() == hf_ids, f"batch mismatch for {text!r}"


def test_superword_roundtrips(superbpe_pair):
    _hf, giga = superbpe_pair
    for text in ["the United States", "machine learning", "a b c", "hello world"]:
        ids = giga.encode(text.encode("utf-8"))
        assert giga.decode(ids) == text.encode("utf-8")
