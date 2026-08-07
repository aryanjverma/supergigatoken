"""Axis 3b - vocabulary differential: our SuperBPE vocab vs the reference's.

``parity.py`` compares the two trainers on *outcomes* -- wall-clock, bytes per
token, superword rate. That leaves the obvious question unanswered: do they
learn the *same tokens*? This script answers it directly, over two
``tokenizer.json`` files trained at matched settings.

    uv run --no-sync benchmarks/superbpe/vocab_diff.py \
        --ours benchmarks/superbpe/artifacts/supergigatoken_s1_100mb.json \
        --reference benchmarks/superbpe/reference/artifacts/reference_superbpe.json

What it reports, and why each number is here rather than a single "overlap":

- **Vocabulary overlap**, whole and split by subword vs superword. Superwords
  are the interesting half: they are what stage 2 exists to learn, and they are
  where two trainers have the most freedom to disagree.
- **Merge-list overlap and rank agreement.** Two vocabularies can hold the same
  tokens and still tokenize differently, because BPE output depends on merge
  *priority*. Spearman correlation over the merges both sides learned says
  whether they agree on order, not just on membership.
- **Rank displacement of shared tokens**, i.e. how far a token moves between
  the two vocabularies. A token learned 5 merges later is a near-match; one
  learned 20000 later is a different decision that happens to land in both.
- **Length and shape distributions**, so a difference in overlap can be read as
  "the reference learns longer superwords" rather than left unexplained.
- **Concrete examples on both sides**, because an aggregate cannot show that one
  trainer learned ``" of the"`` and the other ``" of"`` + ``" the"``.

Nothing here needs the forked ``tokenizers``: it reads the JSON directly and
inverts the GPT-2 byte<->unicode alphabet itself, so it runs in the ordinary
gigatoken env.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common


def gpt2_unicode_to_bytes() -> dict[str, int]:
    """Inverse of the GPT-2 byte<->unicode alphabet a ByteLevel vocab is written in."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


U2B = gpt2_unicode_to_bytes()


def decode_token(tok: str) -> bytes:
    try:
        return bytes(U2B[ch] for ch in tok)
    except KeyError:
        # Added/special tokens are stored literally, not byte-level encoded.
        return tok.encode("utf-8", "replace")


def load(path: Path) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    """``({token bytes: id}, [(left, right), ...])`` from a BPE tokenizer.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data["model"]
    vocab = {decode_token(tok): int(tid) for tok, tid in model["vocab"].items()}
    merges: list[tuple[bytes, bytes]] = []
    for m in model.get("merges", []):
        # tokenizers writes merges as "a b" strings in older formats and as
        # ["a", "b"] pairs in newer ones.
        left, right = m.split(" ", 1) if isinstance(m, str) else (m[0], m[1])
        merges.append((decode_token(left), decode_token(right)))
    return vocab, merges


def is_superword(t: bytes) -> bool:
    """A token that bridges whitespace: an interior space with a non-space before it.

    The same predicate ``run_reference.py`` and ``train_baselines.py`` use, so
    the counts here are comparable to theirs.
    """
    return any(b == 0x20 and i > 0 and t[i - 1] != 0x20 for i, b in enumerate(t))


def spearman(xs: list[int], ys: list[int]) -> float | None:
    """Spearman rank correlation, computed without scipy (ties averaged)."""
    n = len(xs)
    if n < 2:
        return None

    def ranks(vs: list[int]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(num / (dx * dy), 4) if dx and dy else None


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return round(len(a & b) / union, 4) if union else 0.0


def pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def summarize(label: str, vocab: dict[bytes, int]) -> dict:
    supers = [t for t in vocab if is_superword(t)]
    return {
        "label": label,
        "vocab_size": len(vocab),
        "n_superwords": len(supers),
        "superword_fraction": round(len(supers) / max(1, len(vocab)), 4),
        "mean_token_len": round(sum(len(t) for t in vocab) / max(1, len(vocab)), 2),
        "mean_superword_len": round(sum(len(t) for t in supers) / max(1, len(supers)), 2),
        "max_superword_len": max((len(t) for t in supers), default=0),
        "mean_words_per_superword": round(
            sum(t.strip().count(b" ") + 1 for t in supers) / max(1, len(supers)), 2
        ),
    }


def compare(ours: dict[bytes, int], ref: dict[bytes, int], our_merges, ref_merges) -> dict:
    our_set, ref_set = set(ours), set(ref)
    shared = our_set & ref_set
    our_sup = {t for t in our_set if is_superword(t)}
    ref_sup = {t for t in ref_set if is_superword(t)}
    our_sub, ref_sub = our_set - our_sup, ref_set - ref_sup

    # Rank displacement over shared tokens: |id_ours - id_ref|. IDs are merge
    # order, so this is "how much later/earlier each side learned it".
    disp = sorted(abs(ours[t] - ref[t]) for t in shared)
    median_disp = disp[len(disp) // 2] if disp else None

    # Merge-order agreement over the merges both sides learned. A merge's index
    # in the list *is* its priority.
    our_mi = {m: i for i, m in enumerate(our_merges)}
    ref_mi = {m: i for i, m in enumerate(ref_merges)}
    both = [m for m in our_mi if m in ref_mi]
    rho = spearman([our_mi[m] for m in both], [ref_mi[m] for m in both])

    return {
        "vocab": {
            "shared": len(shared),
            "ours_only": len(our_set - ref_set),
            "reference_only": len(ref_set - our_set),
            "jaccard": jaccard(our_set, ref_set),
            "shared_pct_of_ours": pct(len(shared), len(our_set)),
            "shared_pct_of_reference": pct(len(shared), len(ref_set)),
        },
        "subwords": {
            "shared": len(our_sub & ref_sub),
            "jaccard": jaccard(our_sub, ref_sub),
            "shared_pct_of_ours": pct(len(our_sub & ref_sub), len(our_sub)),
        },
        "superwords": {
            "ours": len(our_sup),
            "reference": len(ref_sup),
            "shared": len(our_sup & ref_sup),
            "jaccard": jaccard(our_sup, ref_sup),
            "shared_pct_of_ours": pct(len(our_sup & ref_sup), len(our_sup)),
            "shared_pct_of_reference": pct(len(our_sup & ref_sup), len(ref_sup)),
        },
        "merges": {
            "ours": len(our_merges),
            "reference": len(ref_merges),
            "shared": len(both),
            "jaccard": jaccard(set(our_mi), set(ref_mi)),
            "rank_spearman": rho,
        },
        "shared_token_id_displacement": {
            "median": median_disp,
            "p90": disp[int(0.9 * len(disp))] if disp else None,
            "max": disp[-1] if disp else None,
            "within_100": pct(sum(1 for d in disp if d <= 100), len(disp)),
        },
    }


def examples(ours: dict[bytes, int], ref: dict[bytes, int], n: int) -> dict:
    our_sup = {t for t in ours if is_superword(t)}
    ref_sup = {t for t in ref if is_superword(t)}

    def show(ts, key) -> list[str]:
        return [t.decode("utf-8", "replace") for t in sorted(ts, key=key)[:n]]

    return {
        # Earliest-learned = most frequent, so these are the decisions that
        # matter most for the token count.
        "shared_earliest": show(our_sup & ref_sup, lambda t: ours[t]),
        "ours_only_earliest": show(our_sup - ref_sup, lambda t: ours[t]),
        "reference_only_earliest": show(ref_sup - our_sup, lambda t: ref[t]),
        "ours_only_longest": show(our_sup - ref_sup, lambda t: -len(t)),
        "reference_only_longest": show(ref_sup - our_sup, lambda t: -len(t)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", required=True, help="our tokenizer.json (train_baselines.py output)")
    p.add_argument("--reference", required=True, help="reference tokenizer.json (run_reference.py output)")
    p.add_argument("--examples", type=int, default=10)
    p.add_argument("--out", default="benchmarks/superbpe/results_vocab_diff.json")
    args = p.parse_args()

    ours, our_merges = load(Path(args.ours))
    ref, ref_merges = load(Path(args.reference))

    result = {
        "sides": {
            "ours": summarize("gigatoken train_superbpe", ours) | {"path": args.ours},
            "reference": summarize("original SuperBPE", ref) | {"path": args.reference},
        },
        "differential": compare(ours, ref, our_merges, ref_merges),
        "examples": examples(ours, ref, args.examples),
    }

    d = result["differential"]
    s = result["sides"]
    print(f"  ours:      {s['ours']['vocab_size']} tokens, {s['ours']['n_superwords']} superwords")
    print(f"  reference: {s['reference']['vocab_size']} tokens, {s['reference']['n_superwords']} superwords")
    print(f"  vocab      shared {d['vocab']['shared']}  Jaccard {d['vocab']['jaccard']}")
    print(f"  subwords   shared {d['subwords']['shared']}  Jaccard {d['subwords']['jaccard']}")
    print(f"  superwords shared {d['superwords']['shared']}  Jaccard {d['superwords']['jaccard']}")
    print(f"  merges     shared {d['merges']['shared']}  Spearman {d['merges']['rank_spearman']}")
    disp = d["shared_token_id_displacement"]
    print(f"  shared-token id displacement: median {disp['median']}, p90 {disp['p90']}, {disp['within_100']}% within 100")
    print(f"  shared superwords (earliest):   {result['examples']['shared_earliest'][:5]}")
    print(f"  ours-only superwords (earliest):{result['examples']['ours_only_earliest'][:5]}")
    print(f"  ref-only superwords (earliest): {result['examples']['reference_only_earliest'][:5]}")

    # Through common.save_json so every results file in this suite has one
    # writer, one shape, and the same escaping.
    common.save_json(str(args.out), result)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
