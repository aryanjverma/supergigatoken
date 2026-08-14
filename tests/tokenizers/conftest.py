"""Fixtures for end-to-end tokenizer parity tests against HuggingFace.

Every test in this directory is parametrized over TOKENIZER_SPECS via the
session-scoped `spec` fixture, so adding a tokenizer here adds it to the
whole suite (small-string parity, added-token handling, decode roundtrip,
and the large-scale OWT comparison).

A spec's `name` must have a matching `<name>_tokenizer_path` fixture in
tests/conftest.py that downloads the tokenizer.json.
"""

from dataclasses import dataclass

import pytest
from tokenizers import Tokenizer

from gigatoken.gigatoken_rs import BPETokenizer


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    eot_text: str  # the end-of-text special token
    eot_id: int
    normalizes_nfc: bool = False  # tokenizer.json declares an NFC normalizer
    #: Decoding reproduces the input exactly. False for WordPiece: its
    #: pretokenizer drops whitespace and its normalizer lowercases and strips
    #: accents, so no decoder can invert an encode — HF's own WordPiece decoder
    #: re-inserts spacing heuristically. Such specs skip the roundtrip test and
    #: are covered by test_wordpiece.py instead.
    lossless_decode: bool = True


TOKENIZER_SPECS = {
    s.name: s
    for s in [
        TokenizerSpec(name="gpt2", eot_text="<|endoftext|>", eot_id=50256),
        TokenizerSpec(name="olmo3", eot_text="<|endoftext|>", eot_id=100257),
        TokenizerSpec(
            name="qwen2",
            eot_text="<|endoftext|>",
            eot_id=151643,
            normalizes_nfc=True,
        ),
        TokenizerSpec(
            name="qwen3_5",
            eot_text="<|endoftext|>",
            eot_id=248044,
            normalizes_nfc=True,
        ),
        TokenizerSpec(
            name="modernbert",
            eot_text="<|endoftext|>",
            eot_id=50279,
            normalizes_nfc=True,
        ),
        TokenizerSpec(name="glm5_2", eot_text="<|endoftext|>", eot_id=154820),
        TokenizerSpec(name="deepseek_v3", eot_text="<｜end▁of▁sentence｜>", eot_id=1),
        TokenizerSpec(name="deepseek_v4", eot_text="<｜end▁of▁sentence｜>", eot_id=1),
        TokenizerSpec(name="superbpe_128k", eot_text="<|endoftext|>", eot_id=128000),
        # WordPiece. BERT has no end-of-text token; [SEP] is the closest
        # equivalent and, like every added token, is matched atomically in the
        # raw input, so the eot test still means something.
        TokenizerSpec(
            name="bert_base_uncased",
            eot_text="[SEP]",
            eot_id=102,
            lossless_decode=False,
        ),
        TokenizerSpec(
            name="bert_base_cased",
            eot_text="[SEP]",
            eot_id=102,
            lossless_decode=False,
        ),
        TokenizerSpec(
            name="bert_multilingual",
            eot_text="[SEP]",
            eot_id=102,
            lossless_decode=False,
        ),
    ]
}


@pytest.fixture(scope="session", params=sorted(TOKENIZER_SPECS))
def spec(request) -> TokenizerSpec:
    return TOKENIZER_SPECS[request.param]


@pytest.fixture(scope="session")
def tokenizer_path(spec, request):
    return request.getfixturevalue(f"{spec.name}_tokenizer_path")


@pytest.fixture(scope="session")
def hf_tok(tokenizer_path):
    return Tokenizer.from_file(str(tokenizer_path))


@pytest.fixture(scope="session")
def gigatoken_tok(tokenizer_path):
    return BPETokenizer.from_hf(tokenizer_path)
