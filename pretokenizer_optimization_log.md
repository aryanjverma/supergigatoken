# Pretokenizer Optimization Log

Optimizing the GPT-2 pretokenizer regex:
```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Target: 1 GiB/s single-threaded throughput on 100 MB of OpenWebText.

Platform: Apple Silicon (ARM), `cargo bench` with `lto = "fat"`.

## Baseline

| Implementation | Throughput |
|----------------|-----------|
| `fancy-regex` | ~47 MiB/s |
| State machine (hand-rolled) | ~380 MiB/s |
| Winnow combinators + NEON SIMD | ~462 MiB/s |

The winnow+NEON implementation was the existing best. It uses NEON intrinsics (`vld1q_u8`, `vcgtq_u8`, etc.) to scan 16 bytes at a time inside letter/digit/other runs, with scalar fallback for non-ASCII and run transitions.

---

## Step 1: LUT dispatch + SWAR letter scanning

**File:** `src/pretokenize/pretoken_fast.rs` (new)

**What changed:**
- Replaced the winnow parser combinator framework (`alt()`, `trace()`, `backtrack()`, `ModalResult`) with a direct `Iterator` implementation — zero framework overhead per token.
- Replaced NEON SIMD intrinsics with SWAR (SIMD Within A Register): loads 8 bytes as a `u64`, applies branchless arithmetic to check all 8 bytes for the letter property simultaneously.
- Added a 256-byte LUT (`CLASS[256]`) for O(1) first-byte classification — dispatches directly to the right scan function without cascading `if/else` or `alt()` backtracking.
- Used arithmetic byte predicates instead of LUT lookups inside scan loops: `is_letter(b) = (b | 0x20).wrapping_sub(b'a') < 26`, `is_digit(b) = b.wrapping_sub(b'0') < 10`.

**SWAR letter check (the key technique):**
```rust
let word: u64 = read_unaligned(ptr); // load 8 bytes
let lowered = word | 0x2020_2020_2020_2020; // case-fold
let ge_a = (lowered | 0x8080..).wrapping_sub(0x6161..); // >= 'a'
let le_z = 0xFAFA..wrapping_sub(lowered);               // <= 'z'
let mask = ge_a & le_z & 0x8080..;                       // bit 7 set per letter
// Find first non-letter:
(!mask & HI).to_le().trailing_zeros() / 8
```

**Why it matters:** The SWAR technique processes 8 ASCII letters per iteration with ~6 arithmetic ops — no architecture-specific intrinsics. The LUT dispatch eliminates the alt/backtrack overhead that winnow pays for every token start (trying contraction, then letter_run, then number_run, etc. until one succeeds). Arithmetic predicates avoid data-dependent LUT loads inside hot scan loops.

**Result:** 544 → **830 MiB/s** (first version, without `get_unchecked`)

---

## Step 2: Unsafe `get_unchecked` in scan loops

**What changed:**
- Replaced bounds-checked `bytes[self.pos]` with `unsafe { *bytes.get_unchecked(self.pos) }` in all hot scan loops.
- Used `unsafe { *bytes.get_unchecked(start) }` for first-byte dispatch.

**Why it matters:** Bounds checks in tight loops generate conditional branches that the CPU must predict. Since we always check `self.pos < len` before indexing, the bounds check is redundant — removing it eliminates ~1 branch per byte in letter/digit/other scans.

**Result:** 830 → **840 MiB/s** (~1% improvement, within noise for most runs but consistently measurable)

---

## Step 3: Arithmetic space dispatch (no second LUT lookup)

**What changed:**
- When the first byte is `' '` (space), the second byte determines the token type. Previously this did a second `CLASS[b1]` LUT lookup. Replaced with direct arithmetic checks: `is_letter(b1)`, `is_digit(b1)`, `is_ascii_ws(b1)`, `b1 >= 0x80`.

**Why it matters:** Avoids a data-dependent load (LUT indexed by `b1`) in the most common token start pattern (`" word"`). The arithmetic checks can execute in parallel in the ALU without waiting for the LUT load to complete.

**Result:** No measurable change (~840 MiB/s), but eliminates a potential latency bottleneck on architectures with higher L1 latency.

---

## Step 4: Hot/cold split with `#[cold]` + `#[inline(never)]` (REVERTED)

**What changed:**
- Moved unicode handling (non-ASCII letter/digit/other continuation) into separate `#[cold] #[inline(never)]` functions, keeping only ASCII logic in the hot path.

**Why it matters (in theory):** Reduces the instruction footprint of the hot path, improving icache utilization. The cold functions are rarely called (English text is >99% ASCII).

**Result:** **Regressed to 580 MiB/s.** The `#[inline(never)]` barrier prevented LLVM from optimizing the combined ASCII+unicode loop. The function call overhead (~5 cycles) per unicode encounter was worse than the icache benefit. **Reverted.**

---

## Step 5: `advance()` / `count()` separation

**What changed:**
- Extracted a shared `advance(&mut self)` method that advances `self.pos` past one token without constructing a `Pretoken` slice.
- `next()` calls `advance()` then returns the slice.
- `count()` calls `advance()` in a tight loop, avoiding `Option<Pretoken>` construction.

**Why it matters:** The `count()` hot loop becomes `while pos < len { advance(); n += 1; }` — no `Option` wrapping/unwrapping, no slice construction. This shaves a few nanoseconds per token from the benchmark.

**Result:** ~840 → **848 MiB/s** (small but consistent improvement)

---

## Step 6: Two-pass approach — classification buffer + SWAR transition counting (EXPERIMENTAL)

**What changed:**
- Pass 1: Classify every byte via LUT into a per-byte class buffer (`cb[CHUNK+1]`).
- Pass 2: SWAR XOR adjacent classes to detect transitions; `count_ones()` for the count. Merges WHITESPACE→SPACE and APOSTROPHE→OTHER to suppress false transitions.
- Pass 3: Sequential fixups for whitespace splits (2+ ws followed by non-ws), contractions, and non-ASCII.

**Challenges encountered:**
- SPACE and WHITESPACE are different class values but represent the same merged class for transition purposes — required merging before XOR.
- APOSTROPHE and OTHER needed merging too (non-contraction `'` is scanned as "other").
- Contractions like `c'mon` where the letter after the contraction continues required careful handling — only subtract the APOS→LETTER transition when the byte AFTER the contraction is a different class.
- Non-ASCII bytes needed real unicode classification in Pass 1 to avoid spurious transitions.
- Multi-byte UTF-8 characters spanning chunk boundaries needed chunk-end alignment.

**Why it didn't work:**
- The classification buffer doubles memory traffic (write all classes, then read them back).
- Three passes over the data (classify + SWAR count + fixups) read more total bytes than one-pass.
- The SWAR transition detection saves branch mispredictions but the memory bandwidth cost dominates.

**Result:** **306-354 MiB/s** — 2.5x slower than one-pass. The approach is algorithmically correct (verified on 5 MB) but not competitive for performance.

---

## Step 7: PGO — Profile-Guided Optimization (NO EFFECT)

**What changed:**
- Built with `-Cprofile-generate`, ran the benchmark to collect branch profiles, then rebuilt with `-Cprofile-use`.

**Why it didn't help:**
- The SWAR inner loop is already branchless — no branch probabilities to optimize.
- The word-boundary misprediction is fundamentally unpredictable — the branch outcome depends on input data (is the next byte a letter or not?), not on static code patterns.
- The LUT dispatch compiles to a jump table — PGO can't improve indirect branch prediction.

**Result:** 842 MiB/s (within noise of the 847 MiB/s baseline). PGO actually *hurt* the state machine implementation by 13%.

---

## Step 8: Dual-cursor ILP exploitation

**What changed:**
- Refactored `advance()` from a method into a free function `advance_pos(bytes, pos) -> new_pos` with standalone scan helpers (`scan_letters_from`, `scan_digits_from`, etc.).
- Added `find_split()`: searches for a safe split point near the midpoint — a `\n` followed by a non-whitespace ASCII byte, which guarantees a token boundary.
- Implemented `count_dual_cursor()`: splits the input at the safe boundary, then runs two independent cursors in an interleaved loop:

```rust
while p1 < split && p2 < len {
    p1 = advance_pos(bytes, p1);
    p2 = advance_pos(bytes, p2);
    count += 2;
}
```

**Why it matters:**

The fundamental bottleneck at ~840 MiB/s was **latency, not throughput**. Each token has a serial dependency chain:

```
find_end(token N) → pos_N → load byte at pos_N → classify → scan → find_end(token N+1) → ...
```

This chain is ~25-27 cycles on modern CPUs. The OoO engine has spare execution units sitting idle while waiting for each chain to resolve.

The dual-cursor technique creates two completely independent chains — different positions, different memory addresses, different registers. The OoO engine interleaves their micro-ops across execution ports: while cursor 1 is stalled waiting for a SWAR comparison result, cursor 2's loads and ALU ops execute on the otherwise-idle ports. Even though the two `advance_pos` calls look sequential in source, the CPU sees them as independent instruction streams and overlaps them.

**Result:** 840 → **1,049 MiB/s (1.05 GiB/s)** — 25% speedup, crossing the 1 GiB/s target.

---

## Summary

| Step | Technique | Throughput | Delta |
|------|-----------|-----------|-------|
| Baseline | Winnow + NEON | 462 MiB/s | — |
| 1 | LUT dispatch + SWAR | 830 MiB/s | +1.80x |
| 2 | `get_unchecked` | 840 MiB/s | +1.01x |
| 3 | Arithmetic space dispatch | 840 MiB/s | ~1.00x |
| 4 | Hot/cold split | 580 MiB/s | **reverted** |
| 5 | advance/count separation | 848 MiB/s | +1.01x |
| 6 | Two-pass SWAR transitions | 354 MiB/s | **not used** |
| 7 | PGO | 842 MiB/s | ~1.00x |
| **8** | **Dual-cursor ILP** | **1,049 MiB/s** | **+1.25x** |

**Total speedup over winnow+NEON: 2.27x**
**Total speedup over regex: 22.3x**

Key lessons:
- SWAR is the single biggest win — portable, no intrinsics, processes 8 bytes/iteration.
- Framework overhead (winnow's alt/backtrack) matters more than SIMD width for this workload.
- Multi-pass approaches lose to single-pass due to memory bandwidth, even when branch-free.
- PGO doesn't help when the bottleneck is data-dependent branches.
- ILP exploitation via dual cursors provides free speedup by filling pipeline bubbles.

---

## Addendum: SuperBPE two-level encoding on the released 128k

A different hot path from everything above — the `superword` two-level encoder,
not the pretokenizer scan — recorded here because this is the file CLAUDE.md
points at for perf history. Every figure below is single-threaded, release,
min-of-5, over 33.5 MB of OpenWebText (6819 documents); the CPU is the one
`benchmarks/superbpe/REPORT.md` names in its header. Per
`profiling/campaign_report.md` §2 these are only comparable *within* one
process — the A/Bs below all are.

### The glue rule set: derived, not global

The released 128k's outer regex is `\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)`.
The `{2,}` where the o200k stage-1 family has `+` is the entire hazard class:
`" ’s"` is **one** outer pretoken but **two** stage-1 pretokens (`" ’"`, `"s"`),
so the junction of a low-ID merge falls strictly inside an outer piece and no
amount of outer-awareness can help. With only the whitespace / apostrophe /
digit rules the derived threshold was **485**, pinned by `" ’"`+`"s"` and
continuing `’|t` 682, `-|s` 1269, `’|re` 1549, `.|S` 1920, `" Mc"|"C"`. At 485
level 1 applies almost no merges and two-level encoding loses to the plain path
it exists to beat.

Two further rules fix it — a word-initial (`\p{L}`/`\p{N}`) right side, and a
whitespace-initial right side after a `\p{M}` tail (`("\u{fe0f}", "\n")` alone
caps at 91476, because marks reach the *letter* alternative, which has no
`[\r\n/]*` tail to swallow the newline). Together they lift the threshold to
**13471** — and then a third, unconditional rule (next section) to **85956**,
within 15% of the release's transition point at 100164.

They are not free, and gluing is not monotone in value even though it is
monotone in soundness (it only removes level-1 split points, and removing all of
them *is* the plain path). Coarser units mean rarer level-1 pretoken-cache keys:

| rule set | MB/s | note |
|---|---:|---|
| narrow only | 163.9 | whitespace / apostrophe / digit |
| + the two wide rules | 147.9 | −9.8% |
| + glue every non-word left side | 145.1 | −2.8 more points, **no** further threshold |

So the rule set became a *derived* property. `SuperwordPlan::build_capped`
probes `[(scheme, wide), (scheme, narrow)]` for each candidate scheme and keeps
the highest threshold, ties going to narrow. The released 128k selects wide and
goes 485 → **13471** at this point in the story; the committed 50k artifact
reaches 40000 either way, stays narrow, and re-measured at **163.5 MB/s** with
selection in place — i.e. it pays nothing for a rule it does not need. (13471 is
where wide alone leaves the release. The section below is why it is not the end,
and takes it to 85956.)

Full fill-shape A/B after the change (`bench_superword_variants`, interleaved,
one resident tokenizer): iter+whole 118.9, iter+cuts 141.4, buf+whole 119.2,
buf+cuts 142.7, 2phase+whole 131.6, **2phase+cuts 163.5** (+37.6% over arm 0).

Re-run after the unconditional non-ASCII rule below landed: iter+whole 113.5,
iter+cuts 132.5, buf+whole 116.1, buf+cuts 139.0, 2phase+whole 130.3,
**2phase+cuts 155.3** (+36.9% over arm 0) — every arm ~5% below the run above,
which is what a *machine* difference looks like rather than a rule's cost. The
control is in the next section but one: that session's released-128k **plain**
arm read 31.6 against 33.4, −5.4%, and the plain path never calls `glues` at
all. So the rule's own cost on English prose is below what this box can resolve
between sessions, and the shape of the A/B — which arm wins, by how much — is
unchanged. Reported as two runs rather than one edited table, because the
comparison that matters here is *within* each run.

### The junction test was wrong about character fragments

Encoding the release against HuggingFace over the 99.7 MB eval slice disagreed
by **5 tokens in 16,007,082** — one document out of 19,937, in Arabic. Small
enough that the equivalence tests missed it (their corpus had CJK, Devanagari
and accented Latin, but no Arabic) and the throughput harness reported it only
as a token count in the JSON.

Minimal case: `"ا،"`, four bytes, where we produced `[5438, 48629]` and HF
`[13471, 221]`. The token bytes name the bug:

```
13471 = b'\xd8\xa7\xd8'   221 = b'\x8c'      <- HF
 5438 = b'\xd8\xa7'     48629 = b'\xd8\x8c'  <- ours
```

Merge 13471 is `"ا"` + `b"\xd8"`: a whole Arabic letter plus the **bare lead
byte** of the next character. Byte-level BPE merges pairs of *tokens*, and
tokens are byte strings, so nothing stops one from ending mid-character.
`derive_threshold` asked "does each side decode to a character?" and, on a
`None`, concluded the junction was interior — where no pretokenizer can split
and any merge is therefore safe. That conflates two different situations:

1. the left side ends mid-character and the right side **completes** it, so the
   junction really is inside one character; and
2. the left side ends *on* a character boundary and the right side is a
   truncated head, so the junction **is** a boundary — only the character after
   it is unknown.

Case 2 is merge 13471, whose boundary is real (`b"\xd8"` reaches U+0600–U+063F,
which holds Arabic letters and ARABIC COMMA U+060C alike), so admitting it below
the threshold let level 1 fail to apply it across the letter|comma split. The
`Junction` enum now separates the cases, and the unknown character is
**enumerated** rather than assumed: every continuation-byte completion that is
valid UTF-8 gets probed, and one splittable completion condemns the merge.

Enumeration alone is not enough. `("ا", b"\xd8")`'s completions include the
comma, so it stays hazardous and caps the release at 13471 — and the committed
50k artifact at 24013, on the same merge, which means **this was a live bug in
the shipped artifact too**, not something the 128k work introduced. What
recovers the threshold is a third glue rule, `non_ascii_pair`: if the byte at
either side of the junction is ≥ 0x80, both characters are multi-byte, so glue.
Two comparisons, no decode, correct for every completion of a truncated
character — a byte ≥ 0x80 at a token's edge means that character is multi-byte
whether the byte is a lead or a continuation. It is unconditional rather than
part of `wide` because every multilingual byte-level vocabulary has fragment
merges, and because its cost is bounded by the *corpus* rather than the
tokenizer: it can only fire between two non-ASCII characters, which English
prose does not contain.

That takes the release to **85956**, and the remaining gap to 100164 is one
merge that is genuinely unsafe: `b"\x8a"` + `b"\n"`, a lone continuation byte
joined to a newline. `b"\x8a"` completes to り (`b"\xe3\x82\x8a"`) among others,
and letter-then-newline is a real stage-1 boundary — `wide` glues whitespace
after a combining *mark*, not after a letter. `open_left_cap_is_a_real_boundary`
probes exactly that, so the conservative "left operand opens mid-character ⇒
unsafe" arm is pinned as *exact* on this checkpoint rather than merely safe.

Census of the four junction classes, which is what says the enumeration is
affordable and the conservative arm cheap:

| | released 128k | 50k artifact |
|---|---:|---:|
| merges | 127757 | 49744 |
| determined (probed directly) | 125633 | 49292 |
| glued by `non_ascii_pair` | 2083 | 430 |
| open-right (enumerated) | 38 | 22 |
| open-left (capped) | 3 | 0 |

`non_ascii_pair` runs first, so what reaches the enumeration is only a fragment
right side with an *ASCII* left side — and on both checkpoints that left side is
`b" "` every time, with `missing` never above 2 (≤4096 completions × 64 probe
contexts). All three open-left merges are the same shape, a continuation byte
joined to `b"\n"`: 85956, 89003, 90714. The artifact having none is why its
threshold is unaffected by that arm, and why the arm went unnoticed until the
release was loaded.

So 100164 is the release's semantic transition (the first merge that spans
whitespace by intent) but was never a sound threshold. The lesson is the same
one the module docs already carried one level up: probe the **actual token
bytes**, and treat "I could not decode this" as ignorance, not as licence.
`"ا،"` and three longer Arabic strings are now in `SUPERWORD_CASES`.

### Two-level vs plain on the released checkpoint

`bench_released_128k_vs_plain`, threshold 85956, token streams asserted
identical over all 6819 documents:

| arm | MB/s |
|---|---:|
| two-level | **85.8** |
| plain | 31.6 |

2.72×. 5365888 tokens over 33.5 MB is 6.24 bytes/token. HuggingFace
`tokenizers` reads ~6.1 MB/s on this checkpoint, so two-level is ~14.1× it.

Re-measured after the threshold dropped 100164 → 85956, and the ratio is the
number that survived: the earlier run read 90.6 / 33.4 / 2.71×, and the *plain*
arm moved by as much as the two-level one even though it never consults the
threshold at all. That is between-process variance, not the 14% shorter level-1
prefix. Moving merges from level 1 to level 2 near the top of the table is close
to free, because the merges in question are rare.

Note that `plain` here is *not* the artifact's plain path: this outer scheme
splits, so even the plain arm gets bounded pretokens rather than one
document-long one. 33.4 MB/s against the artifact's 10.7 is that difference, and
it is why 2.71× is a smaller ratio than the artifact's while the absolute
two-level number is lower too (shorter level-2 runs, plus the outer scan).

### Phase split, and the SIMD verdict: no

`bench_released_128k_phases`. The level-2 replay arm is skipped for this scheme
— replaying a document's concatenated level-1 stream in one call merges across
outer pretokens the real path never crosses, so it computes different tokens,
not a different time. `full − level 1` was already the authority.

| arm | MB/s | share of two-level cost |
|---|---:|---:|
| full | 89.6 | — |
| level 1 (outer scan + fill + cached stage-1 encode) | 170.3 | 53% |
| level 2′ (`full − level 1`) | 189.2 | 47% |
| outer scan alone | 578.6 | 15% |

The scalar `superword_bounded` walker is the only stage of this path with no
SIMD scanner behind it, and it is **15%** of end-to-end cost: an infinitely
fast outer scanner caps at **1.18×**, a plausible ~2 GB/s one at ~1.12×. That
is below the bar, and the work is not small — the mask harvest cannot simply be
filtered down to this scheme, because the official boundaries are not a subset
of stage 1's (`"a \t b"` has an official boundary at offset 2 that stage 1's
`\p{N}{1,3}| ?[^\s\p{L}\p{N}]+` never produces), so it would be a new scanner
rather than a filter. Deferred; the level-2 merge remains the lever.

### Why the *parallel* numbers are not the ones to optimize against

`benchmarks/superbpe/throughput.py` is the multi-threaded counterpart (99.7 MB,
19937 docs, min of 9) and is what the README quotes. It puts the released 128k at
**319.4 MB/s** against HuggingFace's 5.7. Its between-process spread is wide
enough to be worth recording, so that a future re-measurement is not mistaken for
a regression — four runs at identical settings on this box:

| arm | run A | run B | run C | run D | max/min |
|---|---:|---:|---:|---:|---:|
| SuperBPE 50k, gigatoken | 618.2 | 745.9 | 828.8 | 744.9 | 1.34× |
| plain BPE 50k, gigatoken | 2290.9 | 2039.1 | 2556.7 | 2342.4 | 1.25× |
| released 128k, gigatoken | — | 347.7 | 441.9 | 319.4 | 1.38× |
| SuperBPE 50k, HF | 6.76 | 6.00 | 6.58 | 6.48 | 1.13× |

`min of 9` bounds variance *inside* a process and does nothing between them. The
asymmetry against HF is the diagnostic rather than an anomaly: gigatoken finishes
the slice in 134 ms where HF needs 15.4 s, so the fast engine's measurement is
~115× shorter and rayon's 8-worker scheduling is a proportionally larger share of
it. Run B was measurably depressed by a `ruff format --check` over 230 files
running concurrently — the mechanism in miniature, and a reminder to leave the
box alone while measuring.

The single-threaded benches earlier in this addendum do *not* show this, which is
why they are the ones A/B decisions are made on: 33.5 MB on one core is a ~370 ms
measurement with no scheduler in the loop, and the `bench_superword_variants`
arms are interleaved within one process on top of that. Quote the parallel figure
for what a user gets; optimize against the single-threaded one.

## Addendum: WordPiece / BERT

Measurements taken while implementing `bpe::wordpiece` + `bpe::bert_normalizer`
and the `bert` pretokenizer scheme. Box: the same Windows machine as the
addendum above, single-threaded arms unless stated.

### The `bert` walker: 408 MiB/s, and why the comparison flatters the others

`cargo bench --bench pretokenize -- fast_scalar`, 100 MB of OWT, criterion
10 samples:

| scheme | throughput |
|---|---:|
| r50k (`fast_scalar`) | 1.794 GiB/s |
| qwen2 | 1.409 GiB/s |
| qwen3_5 | 1.391 GiB/s |
| cl100k | 1.360 GiB/s |
| **bert** | **408.4 MiB/s** |

4.4× behind r50k, but the per-byte comparison is not apples to apples: this
scheme *isolates every punctuation character*, so `"a...b"` is five pretokens
where r50k's is two. A large part of the gap is spans emitted, not bytes walked,
and span count is what the downstream cache probe pays for. The walker itself is
one table load per byte with no SWAR skip — the SIMD `MaskScheme` port is
therefore still open, and unlike `superword_bounded` (15% of its path, capped at
1.18×) this one has not been shown to be a small share of anything. Measure the
share before building it.

### The materialised normalizer cost 2.6× of encode, and the plan said <1%

The design predicted the materialised `BertNormalizer` pass would be "well under
a percent" because "ASCII lowercase is a few GB/s against an encode path running
at hundreds of MB/s". Measured on 32 MB of OWT (best of 5, single-threaded,
32 × 1 MB documents, bert-base-uncased vocab with only the `normalizer` field
varying):

| arm | before | after | share of encode, before → after |
|---|---:|---:|---:|
| no normalizer | 271.0 | 262.5 | — |
| clean_text only | 143.2 | 236.6 | 47.2% → 9.9% |
| clean+cjk | 137.1 | 234.2 | 49.4% → 10.8% |
| clean+cjk+lower | 110.5 | 217.4 | 59.2% → 17.2% |
| clean+cjk+strip (NFD) | 109.9 | 193.7 | 59.5% → 26.2% |
| **full (bert-base-uncased)** | **101.4** | **190.5** | **62.6% → 27.4%** |

End-to-end encode went 101.4 → 190.5 MB/s, **1.88×**, for two changes that are
the same idea applied twice.

Two things the prediction missed, both of which only a decomposition by step
could show:

- **`clean_text` was 47% on its own** — six times the whole normalizer's
  predicted budget — because it rebuilt the document char by char through
  `chars()` + `String::push`, with a class-table load per character. NFD, the
  step that *looks* expensive, was 0.2%.
- **A whole-segment ASCII precheck buys nothing.** The first attempt gated the
  fast path on "printable ASCII and no uppercase" over the entire segment. It
  never fired: uppercase is in nearly every document, and once that was fixed,
  *newlines* still disqualified every 1 MB OWT document. Throughput moved by
  −6% (102.0 → 95.3), i.e. noise plus a wasted scan. The win only exists
  per **run**: hop to the next byte outside 0x20–0x7E, bulk-copy everything
  before it. That is the same scan `PrecompiledCharsmap::normalize_into` already
  uses two modules over, for the same reason.

The fold pass (drop `Mn`, lowercase) needed the identical treatment and gave the
second half of the win: no ASCII char is `Mn`, so an ASCII run never triggers the
filter and the case fold is a byte map. It cannot be skipped outright even when
the clean pass already folded ASCII — NFD *creates* ASCII that never went through
it (`"É"` → `"E"` + U+0301, and the `E` still needs folding) — but folding an
already-folded byte is idempotent, so applying the map unconditionally is both
correct and cheaper than tracking provenance.

Both bulk paths rest on arguments about where an ASCII run can be treated as
opaque (NFD neither decomposes nor reorders an ASCII starter; no ASCII char is
`Mn`; `to_lowercase` maps ASCII to ASCII, so the steps commute with an ASCII case
fold). Per this repo's convention those arguments are not the warrant —
`bert_normalizer_matches_reference_random` is: it fuzzes the run-based
implementation against a straightforward four-pass reference over 24 flag
combinations and 400 rounds each, with input weighted 70% ASCII so the run
boundaries are where the cases land.

### What is left, and the lever that is still deferred

At 27.4% the normalizer is still far above the design's ~1% trigger for the
deferred **raw-keyed, normalize-on-miss** redesign, and this measurement is what
that decision was waiting for. The remaining cost splits roughly: ICU NFD ~9%,
the clean pass ~10%, the fold pass ~6%. The redesign's appeal is structural
rather than incremental — normalization is the only stage that still runs over
100% of input bytes, where the pretoken cache means MaxMatch runs on ~1% of
pretokens — so its ceiling is the whole 27.4%. It also needs its own differential
fuzz (the split points must provably come out the same), which is why it stays
deferred here rather than being attempted alongside everything else.

### Miss-path cost: LinMaxMatch is not indicated

Cold vs warm passes over the same 32 MB in one process, normalizer stripped so the
number isolates the encode engine: 212.7 MB/s on the first pass, 261.6 on the
best later one — a 19% cold penalty that covers *everything* first-touch (pretoken
cache inserts, token-arena growth, page faults), of which naive MaxMatch is only a
part. On warm passes the miss path does not run at all. The LinMaxMatch trie
(Song et al. 2021) therefore has a ceiling well under 19% on a cold corpus and
~0% on a repeated one, against a naive backoff already bounded to
`max_piece_len` probes per position (18 bytes for bert-base-uncased). Not built;
build it only if a bench isolating the miss path contradicts this.

### End-to-end, against HuggingFace

`benchmarks/compare/measure.py`, bert-base-uncased, 100 MB of OWT, one fresh
process per library, 8 cores:

| library | MB/s | Mtok/s | wall |
|---|---:|---:|---:|
| gigatoken | **618.25** | 136.4 | 0.162 s |
| HuggingFace `tokenizers` | 13.91 | 3.05 | 7.191 s |

**44.4×.** The two rows report slightly different token counts (22.06M vs
21.94M, 0.6%) because the harness hands gigatoken the slab as one document and
HF a list split on the separator, which moves the boundaries at document edges;
that asymmetry is how every existing row in `benchmarks/results.json` was
measured too, so the figure is comparable within that table. The BERT repos are
registered in `sweep.py`'s `REPOS` for the next full sweep — this run was
deliberately *not* merged into `results.json`, since folding rows measured under
different conditions into a curated artifact would quietly break the comparison
it exists to make.
