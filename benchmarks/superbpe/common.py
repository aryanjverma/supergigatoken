"""Shared helpers for the SuperBPE evaluation suite.

Everything the three axis scripts (efficiency, throughput, trainer parity)
and the aggregate report need in common:

- corpus handling: slice a big text file into a disjoint train/eval pair at
  UTF-8 boundaries, with a deterministic synthetic fallback so the pipeline
  runs end-to-end even without the OWT download;
- the gigatoken benchmark tokenizer set (mirrors
  ``benchmarks/compare/sweep.py`` ``REPOS``);
- converting gigatoken ``(vocab, merges)`` training output into an HF
  ``tokenizers.Tokenizer`` / ``tokenizer.json`` (the same ByteLevel
  conversion as ``tests/test_superbpe_train.py``), with ``use_regex=False``
  so learned superwords actually fire at encode time;
- bytes/token counting, superword detection/stats, and small JSON helpers.

Run these scripts with ``uv run --no-project`` (they only need ``gigatoken``
plus ``tokenizers``; ``matplotlib`` is optional and only used for plots).
"""

from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(BENCH_DIR)
COMPARE_DIR = os.path.join(BENCH_DIR, "compare")

# The three axes write their raw results next to this file; report.py reads
# them back.
RESULTS_EFFICIENCY = os.path.join(HERE, "results_efficiency.json")
RESULTS_THROUGHPUT = os.path.join(HERE, "results_throughput.json")
RESULTS_PARITY = os.path.join(HERE, "results_parity.json")
REPORT_MD = os.path.join(HERE, "REPORT.md")

# Released reference SuperBPE tokenizer on the Hub (Liu et al., 2025).
RELEASED_SUPERBPE_REPO = "alisawuffles/superbpe-tokenizer-128k"


# --------------------------------------------------------------------------
# gigatoken benchmark tokenizer set
# --------------------------------------------------------------------------

# Mirror of benchmarks/compare/sweep.py REPOS (imported when available so the
# two lists never drift; the literal copy is only a fallback).
_FALLBACK_REPOS = [
    "openai-community/gpt2",
    "answerdotai/ModernBERT-base",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-V3",
    "zai-org/GLM-4.7",
    "openai/gpt-oss-20b",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "allenai/Olmo-3-1025-7B",
    "moonshotai/Kimi-K2-Instruct",
    "microsoft/phi-4",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "codellama/CodeLlama-7b-hf",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "unsloth/gemma-2b",
    "google/gemma-3-4b-it",
]


def benchmark_repos() -> list[str]:
    """The gigatoken benchmark tokenizer repos (from sweep.py if importable)."""
    if COMPARE_DIR not in sys.path:
        sys.path.insert(0, COMPARE_DIR)
    try:
        import sweep  # type: ignore

        return list(sweep.REPOS)
    except Exception:
        return list(_FALLBACK_REPOS)


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

DEFAULT_SEPARATOR = b"<|endoftext|>"

# Frequent multi-word collocations — these become superwords under a lifted
# whitespace restriction (the whole point of the SuperBPE comparison).
_SYNTH_PHRASES = [
    "the United States", "New York", "machine learning", "according to the",
    "the government said", "in order to", "at the same time", "on the other hand",
    "as well as", "a number of", "of the year", "in the world", "climate change",
    "the White House", "the European Union", "artificial intelligence",
    "more than", "such as", "for example", "in addition to", "the fact that",
    "the last year", "the first time", "around the world", "the end of the",
    "the beginning of", "United Kingdom", "San Francisco", "the study found",
    "researchers said that", "the report said", "the company said",
]

# A base content vocabulary; combined with affixes below into a few thousand
# unique surface words so a matched-vocab BPE can actually reach a large
# target (real OWT has huge word diversity; a fixed sentence list saturates).
_SYNTH_ROOTS = (
    "time year people way day man thing woman life child world school state "
    "family student group country problem hand part place case week company "
    "system program question work govern number night point home water "
    "room mother area money story fact month right study book eye job word "
    "business issue side kind head house serve friend father power hour game "
    "line member law car city commun name president team minute idea body "
    "inform back parent face level office door health person art history party "
    "result change morning reason research moment air teacher force educate "
    "market data model policy report economy energy science medicine soft "
    "network compute algorithm nation region culture nature climate future "
    "process value method theory design method object memory signal traffic "
    "vision language machine feature pattern measure sample record global local"
).split()
_SYNTH_PREFIXES = ["", "", "", "re", "un", "in", "pre", "over", "under", "inter", "sub", "non"]
_SYNTH_SUFFIXES = ["", "", "", "s", "ed", "ing", "ly", "er", "ion", "al", "ment", "ness", "able", "ity"]


def _build_word_pool() -> list[str]:
    """A large deterministic surface-form vocabulary (root x prefix x suffix),
    ordered most-frequent-first so Zipfian sampling favors the plain roots."""
    seen: dict[str, None] = {}
    for r in _SYNTH_ROOTS:  # plain roots first (the frequent head of the Zipf)
        seen.setdefault(r, None)
    for suf in _SYNTH_SUFFIXES:
        for r in _SYNTH_ROOTS:
            seen.setdefault(r + suf, None)
    for pre in _SYNTH_PREFIXES:
        for suf in _SYNTH_SUFFIXES:
            for r in _SYNTH_ROOTS:
                seen.setdefault(pre + r + suf, None)
    return list(seen)


_SYNTH_WORDS = _build_word_pool()

_SYNTH_TEMPLATES = [
    "{P}, the {w1} of the {w2} was a {w3} {w4}.",
    "The {w1} said the {w2} would {v} the {w3} {P}.",
    "{P}, researchers found that the {w1} affects the {w2} and the {w3}.",
    "According to the {w1}, the {w2} {v} more than the {w3} last {w4}.",
    "{P} {P}, and the {w1} {v} the {w2} in the {w3}.",
    "The {w1} and the {w2} {v} a {w3} near the {w4} {P}.",
]
_SYNTH_VERBS = "increased reduced changed reported showed created reached measured improved".split()


def add_corpus_args(parser) -> None:
    """Standard corpus flags shared by every axis script."""
    parser.add_argument(
        "--file",
        default=os.environ.get("SUPERBPE_CORPUS", "~/data/owt_train.txt"),
        help="training/eval corpus (default: %(default)s). Download the OWT "
        "slice as in the README; if the file is missing a small synthetic "
        "corpus is generated so the pipeline still runs.",
    )
    parser.add_argument("--train-mb", type=float, default=500.0, help="MB of the corpus used for training (default: %(default)s)")
    parser.add_argument("--eval-mb", type=float, default=100.0, help="MB of held-out corpus for bytes/token + throughput (default: %(default)s)")
    parser.add_argument("--separator", default="<|endoftext|>", help="document separator in the corpus (default: %(default)s)")


def _utf8_boundary(data: bytes, idx: int) -> int:
    """Smallest index >= idx that is not in the middle of a UTF-8 sequence."""
    n = len(data)
    while idx < n and (data[idx] & 0xC0) == 0x80:
        idx += 1
    return min(idx, n)


def synth_corpus(total_mb: float, separator: bytes = DEFAULT_SEPARATOR, seed: int = 0) -> bytes:
    """A deterministic pseudo-natural corpus for when the OWT file is absent.

    Documents are short runs of sampled sentences joined by ``separator`` so
    downstream splitting behaves like the real corpus. Not representative of
    real bytes/token numbers — it exists so the harness is runnable and the
    controlled our-SuperBPE-vs-our-BPE comparison is still meaningful.
    """
    rng = random.Random(seed)
    target = int(total_mb * 1e6)

    n_words = len(_SYNTH_WORDS)
    mean_rank = max(20, n_words // 6)

    def word() -> str:
        # Zipfian-ish: front of the list is far more frequent, but the long
        # tail is still reached so BPE sees real word diversity.
        i = min(int(rng.expovariate(1 / mean_rank)), n_words - 1)
        return _SYNTH_WORDS[i]

    def sentence() -> str:
        t = rng.choice(_SYNTH_TEMPLATES)
        return t.format(
            P=rng.choice(_SYNTH_PHRASES),
            v=rng.choice(_SYNTH_VERBS),
            w1=word(), w2=word(), w3=word(), w4=word(),
        )

    chunks: list[bytes] = []
    size = 0
    while size < target:
        doc = ("\n".join(sentence() for _ in range(rng.randint(4, 25))) + "\n").encode("utf-8")
        chunks.append(doc)
        chunks.append(separator)
        size += len(doc) + len(separator)
    return b"".join(chunks)[:target]


def load_corpus(path: str, train_mb: float, eval_mb: float, separator: bytes = DEFAULT_SEPARATOR) -> tuple[bytes, bytes, bool]:
    """Return ``(train_bytes, eval_bytes, synthetic)``.

    The eval slice immediately follows the train slice (disjoint), both cut
    at UTF-8 boundaries. When the file is missing, a synthetic corpus of the
    combined size is generated and split the same way.
    """
    expanded = os.path.expanduser(path)
    synthetic = not os.path.exists(expanded)
    if synthetic:
        print(f"corpus {expanded!r} not found; generating a {train_mb + eval_mb:.0f} MB synthetic corpus", file=sys.stderr)
        data = synth_corpus(train_mb + eval_mb, separator)
    else:
        want = int((train_mb + eval_mb) * 1e6)
        with open(expanded, "rb") as f:
            data = f.read(want + 8)
        data = data[:_utf8_boundary(data, min(want, len(data)))]

    split = _utf8_boundary(data, min(int(train_mb * 1e6), len(data)))
    train = data[:split]
    eval_end = _utf8_boundary(data, min(split + int(eval_mb * 1e6), len(data)))
    ev = data[split:eval_end]
    if not ev:  # corpus smaller than train_mb+eval_mb: reuse the tail for eval
        ev = train[-min(len(train), int(eval_mb * 1e6)):]
    return train, ev, synthetic


def split_docs(data: bytes, separator: bytes = DEFAULT_SEPARATOR) -> list[str]:
    """Decode and split into documents on ``separator`` (for HF/token counting)."""
    text = data.decode("utf-8", errors="ignore")
    sep = separator.decode("utf-8")
    docs = text.split(sep) if sep else [text]
    return [d for d in docs if d]


# --------------------------------------------------------------------------
# gigatoken (vocab, merges) -> HF tokenizer
# --------------------------------------------------------------------------


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte<->unicode map (same as tests/conftest.py)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _bytes_to_hf_str(byte_to_unicode: dict[int, str], token: bytes) -> str:
    return "".join(byte_to_unicode[b] for b in token)


def to_hf_bpe(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], use_regex: bool = False):
    """Build an HF ``Tokenizer`` from gigatoken output.

    ``use_regex=False`` lifts whitespace pretokenization so superword merges
    fire — required for SuperBPE at inference. Pass ``use_regex=True`` for a
    plain BPE baseline.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    b2u = gpt2_bytes_to_unicode()
    hf_vocab = {_bytes_to_hf_str(b2u, tok): tid for tid, tok in vocab.items()}
    hf_merges = [(_bytes_to_hf_str(b2u, a), _bytes_to_hf_str(b2u, b)) for a, b in merges]
    tok = Tokenizer(models.BPE(vocab=hf_vocab, merges=hf_merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=use_regex)
    tok.decoder = decoders.ByteLevel()
    return tok


def save_hf_tokenizer(tok, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok.save(path)
    return path


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def is_superword(token: bytes) -> bool:
    """True iff the token bridges whitespace (a space with a non-space before
    it). Leading-space subwords like b' the' do not count."""
    return any(b == 0x20 and i > 0 and token[i - 1] != 0x20 for i, b in enumerate(token))


def superword_stats(vocab: dict[int, bytes]) -> dict:
    supers = [tok for tok in vocab.values() if is_superword(tok)]
    examples = sorted(supers, key=len, reverse=True)[:12]
    return {
        "vocab_size": len(vocab),
        "n_superwords": len(supers),
        "superword_fraction": round(len(supers) / max(1, len(vocab)), 4),
        "superword_examples": [t.decode("utf-8", errors="replace") for t in examples],
    }


def count_tokens_hf(tok, docs: list[str], batch: int = 20_000) -> int:
    encode_batch = getattr(tok, "encode_batch_fast", tok.encode_batch)
    total = 0
    for i in range(0, len(docs), batch):
        for enc in encode_batch(docs[i : i + batch]):
            total += len(enc.ids)
    return total


def count_tokens_gigatoken(tok, docs: list[str]) -> int:
    import awkward as ak

    return int(ak.sum(ak.num(tok.encode_batch(docs, parallel=True))))


def bytes_per_token(n_bytes: int, n_tokens: int) -> float:
    return round(n_bytes / n_tokens, 4) if n_tokens else float("inf")


# --------------------------------------------------------------------------
# JSON + CPU label
# --------------------------------------------------------------------------


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def cpu_label() -> str:
    if COMPARE_DIR not in sys.path:
        sys.path.insert(0, COMPARE_DIR)
    try:
        import results  # type: ignore

        return results.cpu_label()
    except Exception:
        import platform

        return f"{platform.processor() or platform.machine() or 'unknown'} ({os.cpu_count()} cores)"
