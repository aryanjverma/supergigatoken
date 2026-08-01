"""SuperBPE (two-stage subword-then-superword BPE) training tests.

Validates the `train_superbpe` API against the plan's acceptance criteria:
- with transition_point == vocab_size it reproduces `train_bpe` exactly
  (stage 2 adds nothing);
- superword tokens (bridging whitespace) appear only at ids >= transition
  point, never in the stage-1 subword vocabulary;
- the resulting vocab/merges encode and decode text correctly.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from gigatoken import train_bpe, train_superbpe

from conftest import gpt2_bytes_to_unicode as bytes_to_unicode

# ---------------------------------------------------------------------------
# Training corpus (~120 KB after repetition) — same style as
# test_bpe_train_compare.py so both cover the same text distribution.
# ---------------------------------------------------------------------------

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "The rain in Spain stays mainly in the plain.",
    "To be, or not to be, that is the question.",
    "All that glitters is not gold.",
    "A journey of a thousand miles begins with a single step.",
    "In the beginning was the Word, and the Word was with God.",
    "It was the best of times, it was the worst of times.",
    "Once upon a time, there was a little girl named Lily.",
    "The sun was shining and the birds were singing.",
    "Tom and his friend went to the park to play.",
    "The the the the the the the the the the",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
]

CORPUS = "\n".join(_SENTENCES * 100)
CORPUS_BYTES = CORPUS.encode("utf-8")
VOCAB_SIZE = 500
TRANSITION_POINT = 400


def _is_superword(token: bytes) -> bool:
    """A token bridges whitespace (is a "superword") iff it contains a space
    with a non-space byte immediately before it. Leading spaces (" the") do
    not count — those are ordinary subword tokens from ByteLevel BPE."""
    return any(
        b == ord(" ") and i > 0 and token[i - 1] != ord(" ")
        for i, b in enumerate(token)
    )


def _to_hf(vocab: dict, merges: list, use_regex: bool = True) -> Tokenizer:
    """Build an HF Tokenizer from gigatoken training output for round-trip
    checks (ByteLevel pretokenizer + decoder; raw ids are re-mapped to HF's
    unicode alphabet). `use_regex=False` lifts whitespace pretokenization so
    superword merges can actually fire, as a SuperBPE tokenizer needs at
    inference."""
    hf_vocab = {bytes_to_unicode(v): k for k, v in vocab.items()}
    hf_merges = [(bytes_to_unicode(a), bytes_to_unicode(b)) for a, b in merges]
    tok = Tokenizer(models.BPE(vocab=hf_vocab, merges=hf_merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=use_regex
    )
    tok.decoder = decoders.ByteLevel()
    return tok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_transition_equals_vocab_reproduces_bpe():
    """With transition_point == vocab_size, stage 2 runs no merges, so the
    output must be byte-for-byte identical to plain train_bpe."""
    bpe_vocab, bpe_merges = train_bpe(CORPUS_BYTES, VOCAB_SIZE, [])
    super_vocab, super_merges = train_superbpe(
        CORPUS_BYTES, VOCAB_SIZE, VOCAB_SIZE, []
    )
    assert super_vocab == bpe_vocab
    assert super_merges == bpe_merges


def test_vocab_size_and_merge_count():
    vocab, merges = train_superbpe(
        CORPUS_BYTES, VOCAB_SIZE, TRANSITION_POINT, []
    )
    assert len(vocab) == VOCAB_SIZE
    assert len(merges) == VOCAB_SIZE - 256
    # Base byte vocabulary is preserved.
    for i in range(256):
        assert vocab[i] == bytes([i])


def test_superwords_only_after_transition():
    """Stage-1 (subword) tokens never bridge whitespace; superwords appear
    only among the stage-2 vocabulary (ids >= transition point)."""
    vocab, _ = train_superbpe(CORPUS_BYTES, VOCAB_SIZE, TRANSITION_POINT, [])

    superword_ids = [tid for tid, tok in vocab.items() if _is_superword(tok)]
    assert superword_ids, "stage 2 learned no superwords"
    for tid in superword_ids:
        assert tid >= TRANSITION_POINT, (
            f"superword token {vocab[tid]!r} has id {tid} < transition "
            f"point {TRANSITION_POINT}"
        )


def test_superbpe_more_efficient_than_bpe():
    """A SuperBPE vocab, encoded with whitespace pretokenization lifted
    (as SuperBPE requires at inference), should encode the corpus in no more
    tokens than a plain BPE vocab of the same size -- superwords bridging
    whitespace add coverage that plain BPE cannot reach."""
    bpe_vocab, bpe_merges = train_bpe(CORPUS_BYTES, VOCAB_SIZE, [])
    super_vocab, super_merges = train_superbpe(
        CORPUS_BYTES, VOCAB_SIZE, TRANSITION_POINT, []
    )
    bpe_tok = _to_hf(bpe_vocab, bpe_merges)
    # Relaxed pretokenization so the learned superwords can actually fire.
    super_tok = _to_hf(super_vocab, super_merges, use_regex=False)

    sample = "\n".join(_SENTENCES)
    n_bpe = len(bpe_tok.encode(sample).ids)
    n_super = len(super_tok.encode(sample).ids)
    assert n_super <= n_bpe, (
        f"SuperBPE ({n_super} tokens) should be at least as efficient as "
        f"BPE ({n_bpe} tokens)"
    )


def test_roundtrip():
    vocab, merges = train_superbpe(CORPUS_BYTES, VOCAB_SIZE, TRANSITION_POINT, [])
    tok = _to_hf(vocab, merges)
    for text in [
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time there was a little girl.",
        "a b c d e f g",
        "1234567890",
    ]:
        assert tok.decode(tok.encode(text).ids) == text
