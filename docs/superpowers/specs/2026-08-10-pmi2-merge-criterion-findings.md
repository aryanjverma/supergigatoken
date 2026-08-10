# PMI² byte-pair merge criterion — measured, rejected

Status: **investigated and dropped, 2026-08-10.** Recorded so it does not get
re-litigated. No code was written.

## The idea

Replace BPE's merge criterion (raw pair frequency) with WordPiece's association
score, numerator squared, over **byte** pairs:

```
score(a, b) = F(a,b)² / (F(a) · F(b))
```

Motivation: the square should stop the association score from penalising common
characters, and operating on bytes keeps the unigram table small.

## Two things worth keeping

**It is exactly Daille's PMI², and it is scale-free.** With `P(x) = F(x)/N`, the
`N` cancels:

```
P(a,b)² / (P(a)·P(b))  =  (F(a,b)²/N²) / (F(a)F(b)/N²)  =  F(a,b)² / (F(a)·F(b))
```

In log space `PMI^k = PMI + (k−1)·log F(a,b)`, so `k=1` is WordPiece's
`F(a,b)/(F(a)F(b))`, `k→∞` is pure frequency (BPE), and `k=2` is a principled
interpolation one log-frequency term away from WordPiece. No corpus-size
normalisation is ever needed.

**But the square removes exactly one factor of the rare-pair bias, not the bias
itself.** For any perfectly correlated pair — `F(a,b) = F(a) = F(b) = f` — the
score is `f²/f² = 1` **regardless of `f`**. A digraph occurring once ties with
one occurring a billion times. Frequency-blindness on the correlated subset is
the criterion's defining behaviour, and every measurement below is downstream
of it.

## Measurements

Greedy trainer, identical inputs: 1.68 MB of this repo's prose + source as a
stand-in corpus (no OWT checked in), 85/15 train/held-out split, GPT-2-style
pretokenization, 2000 merges.

| criterion | bytes/token (held-out) | vs BPE | valid-UTF-8 vocab |
|---|---|---|---|
| BPE (frequency) | **3.169** | — | 99.7% |
| WordPiece `f/(fa·fb)` | 1.017 | −68% | — |
| **PMI² `f²/(fa·fb)`** | **2.254** | **−28.9%** | 90.4% |
| PMI³ | 2.915 | −8.0% | 97.0% |
| PMI⁴ | 3.082 | −2.7% | — |
| PMI², min `f ≥ 20` | 2.539 | −19.9% | 99.3% |
| PMI², min `f ≥ 100` | 2.851 | −10.0% | 99.9% |

PMI²'s first merges are `\x80\x94`, `\xe4\xb8`, `\xe2\x80\x94`, `\xe4\xb8\xad`,
`\xe8\xaa\x9e` — it spends its opening budget completing rare CJK characters and
em-dashes in an English corpus, exactly as the correlated-pair analysis predicts.
BPE's are `\r\n`, `  `, `en`, `er`, ` t`, `in`.

Three attempts to rescue it, since bytes/token is BPE's *own* objective:

1. **Is UTF-8 the confound?** No. On an ASCII-only corpus PMI² still loses
   **25.3%**. Only ~3.6 points of the 28.9% were multi-byte artefacts — the
   criterion is the problem, not byte-level operation.
2. **Does it buy a cleaner vocabulary?** No, the opposite: 90.4% of its learned
   tokens are valid UTF-8 vs BPE's 99.7%. It completes rare characters early
   but leaves more fragments overall, so it does not help with the character-
   fragment problems documented in `src/bpe/superword.rs`.
3. **Does scale-freeness buy stability?** No, the opposite. Vocab Jaccard
   overlap training on 10% vs 100% of the corpus: BPE 78.6%, PMI³ 57.6%,
   **PMI² 37.6%**. Chasing rare correlated pairs makes it maximally sensitive
   to which rare items landed in the sample.

The `k` sweep is monotone across four points: **for compression, this family's
optimum is `k → ∞`, i.e. BPE.** Min-count floors help only by interpolating
toward BPE.

## Two notes on the premise

- The unigram table is 256 rows only at initialisation. After the first merge
  you need counts of *current symbols*, so it grows to `vocab_size` rows
  (`F(c) = F(a,b)`, `F(a) −= F(a,b)`, `F(b) −= F(a,b)`). Still trivial memory.
- The real implementation cost is not memory but **rescoring fan-out**. BPE only
  updates pairs whose *counts* changed — a small local set, which is why
  `run_merges`' `PriorityQueue::change_priority_by` loop works. Under PMI^k,
  changing `F(a)` changes the score of *every pair containing `a`*, so a
  `symbol → set<pair>` index and a rescore of both operands' pair sets is
  required per merge. Lazy pop-and-revalidate is **not** sound: PMI^k scores can
  *increase* (both `F(a)` and `F(b)` shrink), so a stored score is not an upper
  bound.

## What would revive it

One objective was never tested, and it is the only one where this plausibly
wins: **cross-language equity**. PMI² allocated vocabulary to CJK inside an
English corpus, which under a worst-language or variance-of-per-language
bytes/token objective — the "tokenization tax" on low-resource languages — is
the desired behaviour, where BPE's proportional-to-corpus-share allocation is
not. Testing it needs a multilingual corpus, which the ASCII-heavy repo-text
probe could not provide.

If that is ever picked up: measure **per-language** bytes/token plus
worst-language and coefficient of variation, at vocab 8k–16k, before writing any
Rust. If it does not beat BPE there, the criterion is finished.
