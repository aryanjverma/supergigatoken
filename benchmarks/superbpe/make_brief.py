"""Render the supergigatoken technical report as a .docx, in paper form.

A standalone document generator, not part of the benchmark pipeline. It is laid
out the way a short paper is -- abstract, introduction, two measured sections
each taking process -> experimental setup -> results, limitations, conclusion,
works cited -- and restates the measured numbers from ``REPORT.md`` (plus
``results_trainer.json``, which REPORT.md does not cover) alongside the
published SuperBPE results (Liu et al., 2025).

Numbers are hardcoded rather than read from ``results_*.json``, so the document
builds on a machine that has never run the suite. The cost of that choice is
staleness, so the measurement commit and date are stamped into the document --
see ``MEASURED_COMMIT`` below, and update it whenever these values are refreshed.

Measured figures and quoted figures are kept in separate sections and separate
tables on purpose, so a reader can tell which is which. Attribution is likewise
explicit throughout: the subword encoder and ``train_bpe`` are upstream
gigatoken's, and the report says so wherever their numbers appear.

    uv run --no-project --with python-docx benchmarks/superbpe/make_brief.py
"""

from __future__ import annotations

import argparse
import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

BLUE = RGBColor(0x1D, 0x4E, 0xD8)
GRAY = RGBColor(0x4B, 0x55, 0x63)

# The tree the hardcoded numbers below were measured on. Bump these whenever the
# numbers are re-hardcoded from a fresh benchmark run.
#
# Two commits, not one, and the distinction is load-bearing rather than
# bookkeeping. The throughput figures were re-measured at MEASURED_COMMIT, which
# fixed a junction bug in the two-level threshold derivation -- so the released
# 128k's numbers before it were taken from an encoder that mis-encoded 5 tokens
# in 16M, and its throughput row did not exist at all. The training, efficiency,
# parity and vocabulary-differential figures were not re-run and still belong to
# TRAINED_COMMIT; nothing in the fix touches the trainer. Efficiency is quoted to
# 3 significant figures and the 5-token correction does not move any digit of it
# (6.2311 -> 6.2310 bytes/token), so that table carries across unchanged.
MEASURED_COMMIT = "8c7e092"
MEASURED_DATE = "8 August 2026"
TRAINED_COMMIT = "e1586e9"

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "..", "assets")

# Table and figure numbers. The document is emitted front to back in a single
# pass and never refers forward, so a running counter is all the numbering needs
# -- no second pass to resolve references.
_SEQ = {"table": 0, "figure": 0}


def _next(kind: str) -> int:
    _SEQ[kind] += 1
    return _SEQ[kind]


def _style(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    # 10pt, not 10.5: the paper apparatus (abstract, captions, references) costs
    # most of a page over the plain-brief layout this replaced, and half a point
    # buys it back without making the body harder to read than a conference PDF.
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(4)
    # Justified body text is what makes the page read as a paper rather than a
    # web page. Single-line headings are unaffected (a justified paragraph's last
    # line is left-aligned), and captions override the alignment themselves.
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    # The default template leaves line spacing at Word's 1.08 and the side
    # margins at 1.25in, which costs half an inch of column width -- enough that
    # a full-width figure would run into the right margin. Pin both.
    normal.paragraph_format.line_spacing = 1.0
    for section in doc.sections:
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
    # 24pt before an H1 is a lot for a document this dense; the colour change
    # already separates sections. H2 is pinned rather than left to the template
    # so the spacing is the same whichever Word template renders it.
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(11)
    doc.styles["Heading 2"].paragraph_format.space_before = Pt(7)


def _h(doc: Document, text: str, level: int) -> None:
    p = doc.add_heading(text, level=level)
    # Headings must opt out of the body's justification: a heading long enough to
    # wrap gets its first line stretched to the full column, which is the one
    # place justified text looks obviously wrong.
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.color.rgb = BLUE if level <= 1 else GRAY


def _para(doc: Document, text: str, style: str | None = None):
    """Add a body paragraph, rendering ``**...**`` spans as bold runs.

    Emphasis has to be per-run in docx, and building the runs at each call site
    would bury the prose in markup for the one thing it needs: the headline
    numbers, which a reader should be able to find without reading the sentence
    around them. Splitting on the marker leaves the source readable -- odd
    chunks are the delimited ones, so they are the bold ones.
    """
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    for i, chunk in enumerate(text.split("**")):
        if chunk:
            p.add_run(chunk).bold = i % 2 == 1
    return p


def _caption(doc: Document, text: str, center: bool = True, keep_with_next: bool = False) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    # A caption belongs to the thing it labels, so it takes the body paragraph
    # gap on one side only. Eleven of these at the full 4pt cost most of an inch.
    p.paragraph_format.space_after = Pt(1)
    # Table captions sit *above* their table, and the header row already keeps with
    # its first data row -- so the caption is the one piece left free to strand at
    # the foot of a page while the table it names starts on the next. Figure
    # captions sit below and must not take this: it would tie them to the following
    # body paragraph and let them drift off the page away from their own figure.
    p.paragraph_format.keep_with_next = keep_with_next
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = GRAY


def _note(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = GRAY


# A value and its unit are one word typographically, and the numeric columns are
# narrow enough that Word will otherwise break "10.4 s" across two lines and leave
# the "s" stranded. Glue them with a non-breaking space at render time so the
# table data stays readable as plain strings in the source.
_UNIT = re.compile(r"(\d) (s|MB/s|GB/s|bytes/token|bytes|ids)\b")


def _glue_units(text: str) -> str:
    return _UNIT.sub("\\1\u00a0\\2", text)


def _table(doc: Document, caption: str, headers: list[str], rows: list[list[str]], bold_col0: bool = False) -> None:
    # Table captions go above the table and figure captions below, per the usual
    # convention; both are numbered from the same single pass.
    _caption(doc, f"Table {_next('table')}. {caption}", center=False, keep_with_next=True)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, head in zip(t.rows[0].cells, headers):
        cell.text = ""
        # Cells inherit Normal's justification, which in a narrow column stretches
        # a wrapped label until the words are pages apart ("Subword    encoder:").
        # Every cell therefore states its alignment: labels left, numbers right.
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        # A header row alone at the foot of a page is worse than a table that
        # breaks one row later, so it keeps with the first data row.
        cell.paragraphs[0].paragraph_format.keep_with_next = True
        run = cell.paragraphs[0].add_run(head)
        run.bold = True
        run.font.size = Pt(8.5)
    for row in rows:
        cells = t.add_row().cells
        for i, (cell, val) in enumerate(zip(cells, row)):
            cell.text = ""
            run = cell.paragraphs[0].add_run(_glue_units(val))
            run.font.size = Pt(8.5)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            if bold_col0 and i == 0:
                run.bold = True
    # Word will run a table straight into the following paragraph without a
    # separator, but a full-size empty one costs a line eight times over; a 4pt
    # run with its own trailing gap removed is enough to break them apart, and
    # the following paragraph's own space_before is not what does it.
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(0)
    spacer.add_run("").font.size = Pt(4)


def _figure(doc: Document, filename: str, caption: str, width_in: float = 5.7) -> None:
    """Embed one of the branded figures from ``assets/``, with a caption.

    ``width_in`` exists because the figures are not the same shape: the
    multi-panel ones are wide strips that want the full text column, while the
    single-comparison throughput chart is nearly square and would eat half a page
    at that width. Height follows from the aspect ratio, so this is the only knob.

    Missing figures are skipped with a warning rather than raising: the PNGs are
    generated by ``plot_readme.py`` from benchmark output, so a fresh clone that
    has never run the suite should still be able to build the document. The
    figure counter is only consumed when the picture actually lands, so the
    numbering stays contiguous either way.
    """
    path = os.path.join(ASSETS, filename)
    if not os.path.exists(path):
        print(f"warning: {filename} not found in assets/, skipping the figure")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width_in))
    _caption(doc, f"Figure {_next('figure')}. {caption}")


def build(path: str) -> None:
    doc = Document()
    _style(doc)
    _SEQ.update(table=0, figure=0)

    title = doc.add_heading("Fast SuperBPE: training and encoding superword tokenizers in Rust", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in title.runs:
        run.font.color.rgb = BLUE
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = sub.add_run("A technical report on supergigatoken, a fork of the gigatoken tokenizer adding native SuperBPE training and a fast SuperBPE encoder.")
    r.italic = True
    r.font.color.rgb = GRAY
    _note(
        doc,
        f"Encoding measured at commit {MEASURED_COMMIT}, {MEASURED_DATE}; training and efficiency figures at "
        f"{TRAINED_COMMIT}, unchanged since. OpenWebText, Intel 8-core desktop CPU.",
    )

    # ------------------------------------------------------------------
    _h(doc, "Abstract", 1)
    abstract = _para(
        doc,
        "SuperBPE (Liu et al., 2025) trains byte-pair encoding in two stages, lifting the whitespace restriction "
        "partway through so that later merges learn superwords spanning several words. The resulting tokenizer "
        "encodes the same text in fewer tokens at the same vocabulary size, which at fixed training compute buys "
        "downstream model quality and not merely smaller files. Two costs have stood in the way of using it: "
        "stage-2 training is slow and memory-hungry, and the exported tokenizer declares no pretokenization at "
        "all, which defeats every mechanism a fast subword encoder relies on. This report measures both, in a "
        "Rust implementation built as a strict superset of the gigatoken tokenizer (Rød, 2026). At a matched "
        "50,000-token vocabulary on OpenWebText, train_superbpe completes in **28.1 s** against the reference "
        "implementation's **223.4 s** — **8.0× faster** — while learning a near-identical tokenizer: **47,742 of "
        "50,000** vocabulary entries shared, and merge-order agreement of **Spearman 0.999** over the 47,226 "
        "merges both learned. It also trains at corpus sizes where the reference exhausts system memory. For "
        "encoding, a two-level scheme derived from the trained merge table recovers most of the lost cache reuse, "
        "reaching **730.0 MB/s** against HuggingFace's **6.3 MB/s** on the same tokenizer — **116× faster** — with "
        "bit-identical output. The same derivation applies to the released 128k SuperBPE checkpoint, which it "
        "encodes at **310.6 MB/s** against HuggingFace's **4.9 MB/s** — **63× faster** — agreeing with it exactly, "
        "16,007,082 tokens on both engines. The tokenizer itself reaches **5.67 bytes/token**, **20.7% fewer tokens** than "
        "plain BPE trained on the same corpus at the same vocabulary size, and is more efficient than all 27 "
        "production tokenizers measured, several of which have four to five times the vocabulary. Ordinary BPE "
        "training, inherited from gigatoken, is measured against the HuggingFace trainer here for the first time: "
        "**2.8 s** against **10.4 s**, **3.8× faster**.",
    )
    # Indent the abstract on both sides -- the standard visual cue that it is a
    # summary of the document rather than its first section.
    abstract.paragraph_format.left_indent = Inches(0.3)
    abstract.paragraph_format.right_indent = Inches(0.3)

    # ==================================================================
    _h(doc, "1. Introduction", 1)

    _h(doc, "1.1 Background: byte-pair encoding, SuperBPE, and bytes per token", 2)
    _para(
        doc,
        "Byte-pair encoding (Gage, 1994; Sennrich et al., 2016) builds a tokenizer by starting from raw bytes and "
        "repeatedly merging the most frequent adjacent pair of symbols, recording the order in which it did so. "
        "Before any of that, a pretokenizer splits the corpus into units at whitespace and punctuation, and no "
        "merge may cross a unit boundary — which is why no ordinary BPE token contains a space. Modern byte-level "
        "variants (Radford et al., 2019) keep that restriction and are the near-universal default.",
    )
    _para(
        doc,
        "SuperBPE removes it, in two stages. Stage 1 is ordinary whitespace-pretokenized BPE, producing subwords. "
        "Stage 2 resumes from stage 1's vocabulary with the pretoken boundaries removed, so the remaining merges "
        "learn superwords bridging several words — “ of the”, “ in the United States”, “Story continues below "
        "advertisement”. One token now covers more text, so the same corpus needs fewer tokens at the same "
        "vocabulary size.",
    )
    _para(
        doc,
        "The metric for that is **bytes per token**: total corpus bytes divided by the tokens emitted for it. It "
        "reads like a compression statistic, but at fixed training compute it is a quality lever, because fewer "
        "tokens per document means fewer sequence positions and therefore fewer FLOPs per byte of text both "
        "trained and served. The SuperBPE authors pretrained 8-billion-parameter models from scratch with model "
        "size, vocabulary size (200k) and training compute (17.2 × 10²¹ FLOPs, roughly 330B tokens) all held "
        "fixed, varying only the tokenizer.",
    )
    _table(
        doc,
        "Downstream effect of the tokenizer at fixed compute, from Liu et al. (2025). Quoted from the paper; not measured here.",
        ["Metric (Liu et al., 2025 — not measured here)", "BPE", "SuperBPE", "Delta"],
        [
            ["Average over 30 downstream tasks", "39.8", "43.8", "+4.0"],
            ["MMLU (multiple choice)", "36.5", "44.7", "+8.2"],
            ["Encoding efficiency, bytes/token at 200k vocab", "4.46", "6.63", "+48.7%"],
            ["Inference compute, 10⁹ FLOPs per byte", "3.75", "2.65", "−27%"],
            ["Tasks improved", "—", "25 of 30", "—"],
        ],
        bold_col0=True,
    )
    _para(
        doc,
        "Two details carry forward. First, the efficiency gain is not something a larger vocabulary can buy: "
        "ordinary BPE cannot exceed roughly **4.68 bytes/token** on this kind of data, because that is the mean "
        "length of a whitespace-delimited word and a token may not cross a space. Second, the downstream win is "
        "broad rather than a single-benchmark artifact — 25 of 30 tasks improve — with one statistically "
        "significant regression (LAMBADA, 77.0 to 70.6).",
    )

    _h(doc, "1.2 Contributions, and what is inherited", 2)
    _para(
        doc,
        "supergigatoken trains and encodes both kinds of tokenizer in Rust. It is a fork of gigatoken and a "
        "strict superset of it — same “gigatoken” import name, same CLI — which means the fast subword machinery "
        "is inherited rather than written here. That distinction matters for reading every number that follows:",
    )
    _table(
        doc,
        "Provenance of each component. The fork's contribution is the SuperBPE half; the speed it is built on is not.",
        ["Component", "Origin"],
        [
            ["Subword encoder: SIMD pretokenization, pretoken cache, GB/s byte-level BPE", "upstream gigatoken"],
            ["train_bpe — the ordinary BPE trainer", "upstream gigatoken (first measured against HuggingFace here)"],
            ["train_superbpe — the two-stage SuperBPE trainer", "this fork"],
            ["Superword two-level encoder, and the loader that installs it", "this fork"],
            ["superbpe_stage1 — the original SuperBPE stage-1 pretokenizer scheme", "this fork"],
            ["benchmarks/superbpe/ — the evaluation suite behind this report", "this fork"],
        ],
        bold_col0=True,
    )
    _para(doc, "Where a number below measures gigatoken's own code rather than the fork's, it says so in place.")

    _h(doc, "1.3 Summary of results", 2)
    _para(
        doc,
        "All measurements are on OpenWebText (Gokaslan & Cohen, 2019) at a 50,000-token vocabulary with the "
        "stage-1/stage-2 transition at 40,000. Encoding is measured on a **99.74 MB** held-out slice of **19,937** "
        "documents, disjoint from anything trained on.",
    )
    _table(
        doc,
        "The six headline results, each developed in §2.3 or §3.3.",
        ["Result", "supergigatoken", "Compared against", "Ratio"],
        [
            ["SuperBPE training, 100 MB", "28.1 s", "original SuperBPE, 223.4 s", "8.0×"],
            ["BPE training, 100 MB (gigatoken's trainer)", "2.8 s", "HuggingFace BpeTrainer, 10.4 s", "3.8×"],
            ["Encoding efficiency, matched 50k vocab", "5.67 bytes/token", "plain BPE, 4.49 bytes/token", "20.7% fewer tokens"],
            ["SuperBPE encoding throughput", "730.0 MB/s", "HuggingFace tokenizers, 6.3 MB/s", "116×"],
            ["Released 128k SuperBPE encoding throughput", "310.6 MB/s", "HuggingFace tokenizers, 4.9 MB/s", "63×"],
            ["Subword encoding throughput (gigatoken's encoder)", "2297.1 MB/s", "HuggingFace tokenizers, 4.0 MB/s", "577×"],
        ],
        bold_col0=True,
    )
    _figure(
        doc,
        "superbpe_vs_original.png",
        "The SuperBPE trainer against the only other implementation of it. Same corpus slice, same vocabulary "
        "size, same transition point, same stage-1 regex — so the only thing that differs is the implementation.",
    )

    # ==================================================================
    _h(doc, "2. Training", 1)

    _h(doc, "2.1 Process", 2)
    _para(
        doc,
        "A BPE trainer does not work over corpus text directly. It pretokenizes once, counts how often each "
        "distinct unit occurs, and runs the merge loop over that table of unique units weighted by count. So "
        "training cost tracks the number of distinct units, not bytes, and distinct words grow sublinearly in "
        "corpus size — which is why five times the corpus is nowhere near five times the work: 100 MB trains in "
        "**2.8 s** and 500 MB in **7.0 s**.",
    )
    _para(
        doc,
        "Stage 2 breaks that economy deliberately. With pretoken boundaries removed, the unit is no longer a word "
        "but a span that may contain whitespace, so both the number of distinct units and their length grow "
        "sharply and pair counts must be maintained over much longer sequences. Hence stage 2 dominates both "
        "implementations — 26.1 of our 28.1 seconds, 213.1 of the reference's 223.4, so **93%** and **95%** of "
        "their wall-clocks. This is inherent to the method rather than to either implementation; Liu et al. report "
        "the same asymmetry, noting that stage 2 “requires more system memory and CPU time”. supergigatoken's "
        "stage 2 is linear in unit length, with max_unit_len bounding how long a unit may get.",
    )
    _para(
        doc,
        "One stage-2 choice has a visible consequence later: units are bounded at line breaks, so a superword can "
        "never span a newline. This is why §2.3 is an outcome comparison — same speed class, same quality, nearly "
        "the same vocabulary — and not a claim of byte-identical merges. The reference learns 48 superwords "
        "containing a newline; supergigatoken learns zero, by construction.",
    )
    _para(
        doc,
        "The stage-1 pretokenizer is also a real choice. The original SuperBPE stage-1 regex is available as the "
        "superbpe_stage1 scheme; the trainer's default stays gpt2 so published benchmark numbers keep "
        "reproducing. They are not interchangeable — GPT-2's letter class excludes combining marks, so any script "
        "writing vowels as marks fragments: Devanagari हिन्दी becomes six pretokens, no consonant-plus-matra unit "
        "can ever be learned, and at a 4k vocabulary that costs **44.75%** of bytes per token for Hindi. It also "
        "inflates the apparent superword gain, since stage 2 repairs that damage in the SuperBPE arm only — which "
        "is why the comparison below runs the reference's own regex on both sides.",
    )

    _h(doc, "2.2 Experimental setup", 2)
    _para(
        doc,
        "Both trainer experiments use a 100 MB OpenWebText training slice cut at UTF-8 and document boundaries, "
        "at the vocabulary and transition point given in §1.3 — which yields roughly 8,000 superwords.",
    )
    _para(
        doc,
        "For the SuperBPE comparison, both sides run at the same vocabulary and transition point on the same "
        "slice using the reference implementation's own stage-1 regex — a trainer comparison that differs in "
        "pretokenization is not controlled, for the reason in §2.1. The reference runs unmodified in an isolated "
        "virtual environment, so neither installation influences the other.",
    )
    _para(
        doc,
        "For the BPE comparison against HuggingFace (Moi & Patry, 2023) the controls matter more, because the "
        "obvious version of this benchmark is unfair by default. Both sides are byte-level with the GPT-2 split "
        "regex. HuggingFace is seeded with the full 256-byte initial_alphabet, since otherwise it only learns the "
        "bytes its corpus happens to contain and finishes a smaller search. min_frequency is 0 on both sides so "
        "neither prunes work the other must complete, and both land on exactly 50,000 tokens, so this is equal "
        "work rather than an early exit. Each engine runs in its own process, timing wraps the train call only, "
        "and the figure reported is the minimum of three repeats.",
    )
    _para(
        doc,
        "One deviation is worth naming. The tokenizer used for §3 is trained on 500 MB, but the trainer comparison "
        "runs at 100 MB because the reference implementation does not fit: it reached **21 GB** resident on the "
        "500 MB corpus and was still climbing. That is an observation rather than a measurement, since the run was "
        "abandoned rather than completed.",
    )

    _h(doc, "2.3 Results", 2)
    _para(doc, "First the ordinary BPE trainer — gigatoken's code, not the fork's, but nothing had previously measured it against the trainer nearly everyone uses:")
    _table(
        doc,
        "Ordinary BPE training. gigatoken's trainer against HuggingFace BpeTrainer; both produce exactly 50,000 tokens.",
        ["Trainer, 100 MB to a 50k vocabulary", "Train time", "Throughput"],
        [
            ["gt.train_bpe (gigatoken's trainer)", "2.8 s", "36.0 MB/s"],
            ["HuggingFace tokenizers BpeTrainer", "10.4 s", "9.6 MB/s"],
        ],
        bold_col0=True,
    )
    _para(
        doc,
        "**3.8× faster** at identical output size. tiktoken (OpenAI, 2022) is absent because it ships no trainer "
        "at all. Scaling behaves as §2.1 predicts: the full 500 MB corpus trains a 50k vocabulary in 7.0 seconds, "
        "not the 14 a linear model would suggest. Then the two-stage SuperBPE trainer, which is the fork's:",
    )
    _table(
        doc,
        "Two-stage SuperBPE training at matched settings, with the reference implementation's own stage-1 regex on both sides.",
        ["Trainer, 100 MB at 50k / 40k", "Total", "Stage 1", "Stage 2", "Superwords", "Bytes/token"],
        [
            ["supergigatoken train_superbpe", "28.1 s", "2.0 s", "26.1 s", "8118 (16.2%)", "5.85"],
            ["original SuperBPE", "223.4 s", "10.3 s", "213.1 s", "8894 (17.8%)", "5.65"],
        ],
        bold_col0=True,
    )
    _para(
        doc,
        "**8.0× faster** overall, and marginally more efficient rather than trading quality for speed — 5.85 "
        "against 5.65 bytes/token, measured on the same held-out text with both tokenizers. At the project's "
        "default 500 MB training size, which the reference cannot reach, train_superbpe produces a 50k SuperBPE "
        "with 7,944 superwords in roughly **13 minutes**. Speed is only interesting if the output is the "
        "tokenizer the method validated, so the two vocabularies were compared directly:",
    )
    _table(
        doc,
        "Agreement between the two independently trained tokenizers.",
        ["Agreement between the two learned tokenizers", "Value"],
        [
            ["Shared vocabulary entries", "47742 of 50000 (Jaccard 0.914)"],
            ["Shared subwords", "40370 (Jaccard 0.947)"],
            ["Shared superwords", "7372 (Jaccard 0.765)"],
            ["Merge-order agreement over the 47226 shared merges (Spearman)", "0.999"],
            ["Median token-ID displacement of a shared token", "38 ids (82.7% within 100, p90 492)"],
            ["Mean superword length, ours vs reference", "9.65 vs 9.99 bytes"],
        ],
        bold_col0=True,
    )
    _para(
        doc,
        "The Spearman figure is load-bearing, because BPE output depends on merge priority and not only on which "
        "tokens exist: two tokenizers can share a vocabulary entirely and still segment text differently if they "
        "learned it in a different order. At **0.999**, with shared tokens 38 IDs apart at the median, a shared "
        "token is usually the same decision rather than a coincidence. Divergence concentrates where stage 2 has "
        "the most freedom — superwords, Jaccard 0.765 — and is partly structural rather than stochastic, being "
        "the newline bound from §2.1.",
    )

    # ==================================================================
    _h(doc, "3. Encoding", 1)

    _h(doc, "3.1 Process", 2)
    _para(
        doc,
        "Fast subword encoding, inherited unchanged from gigatoken, rests on two mechanisms. A SIMD scanner finds "
        "pretoken boundaries many bytes at a time instead of running a regex engine over the text, and a pretoken "
        "cache maps a word's bytes straight to its finished token sequence. The cache is the larger effect, and it "
        "works for a distributional reason: word frequencies are heavily skewed, so the same few thousand words "
        "recur constantly and are looked up rather than merged.",
    )
    _para(
        doc,
        "A SuperBPE tokenizer defeats both. Because a superword may contain whitespace, the exported tokenizer "
        "declares no pretokenization at all, so every document arrives as one enormous pretoken. The cache never "
        "hits, because whole documents do not recur; the boundary scanners have nothing to find; and the merge "
        "loop runs a priority queue over thousands of byte symbols per document rather than the handful in a word. "
        "This is why HuggingFace encodes SuperBPE at single-digit MB/s — not a defect in its implementation, but "
        "the shape of the problem it is handed.",
    )
    _para(
        doc,
        "Most of the loss is recoverable, and the fork's Superword encoder recovers it by exploiting how the "
        "tokenizer was trained. Merge priority is the token ID and stage-2 merges are appended after stage-1's, "
        "so the merge table splits at a threshold: below it are merges that cannot span a stage-1 pretoken "
        "boundary, above it the superword merges that can. Encoding then runs in two levels. **Level 1** splits "
        "the document at stage-1 boundaries and encodes each unit through the ordinary cached path using only the "
        "sub-threshold merges — cache reuse, seeded vocabulary and SIMD scanning all intact. **Level 2** runs the "
        "full table over the resulting token stream, about **4.5× shorter** than the byte sequence, and need only "
        "consider boundaries some merge can actually cross. On this corpus that eliminates **51%** of them and "
        "leaves independent runs averaging two symbols, so the priority queue all but disappears.",
    )
    _para(
        doc,
        "Deriving that threshold soundly is the subtle part, and three of its requirements were found by measurement "
        "rather than reasoning. It must come from a merge's junction — the seam between its two operands — not "
        "from whether the whole merged string survives pretokenization, because byte-level BPE produces character "
        "fragments and asking whether a fragment splits describes input that cannot occur. The junction must also "
        "be probed with the actual token bytes against the glued splitter, since whether a position is a boundary "
        "depends on how the surrounding run started.",
    )
    _para(
        doc,
        "The third is the one that cost a correctness bug, and it is about those character fragments. A merge's "
        "operands need not be whole characters, so the junction test has to distinguish two situations that look "
        "identical to a decoder: the left operand ends mid-character and the right one **completes** it, in which "
        "case the junction really is inside a character and no pretokenizer can split there; or the left operand "
        "ends **on** a character boundary and the right one begins with a truncated head, in which case the "
        "junction is a real boundary and only the character after it is unknown. Treating the second as interior "
        "admits a merge that level 1 must apply and does not. Merge 13471 of the released checkpoint is exactly "
        "that shape — a whole Arabic letter joined to a bare lead byte — which made an Arabic letter followed by "
        "ARABIC COMMA encode wrongly. The resolution is not to decode: a byte ≥ 0x80 on both sides of a junction "
        "means both characters are multi-byte whatever the completion turns out to be, so two byte comparisons "
        "settle those junctions soundly, and the truncated heads that remain are enumerated over their possible "
        "completions. Without that rule the checkpoint's threshold falls to 13471 and most of the speed goes with "
        "it; with it the threshold is **85956**, within 15% of the checkpoint's own stage-1/stage-2 transition.",
    )
    _para(
        doc,
        "Three correctness properties close the design. Output is bit-identical to feeding the whole document to the "
        "byte-level merge loop, asserted token for token across the entire evaluation slice rather than on "
        "hand-picked cases. On the released checkpoint, where an independent implementation exists to disagree with, "
        "both engines emit the same **16,007,082** tokens over all 19,937 documents, with no document differing — "
        "the check that caught the fragment bug above, when they differed by 5 tokens. And when the derivation "
        "cannot verify a tokenizer, no two-level plan is installed and the plain path runs — so a tokenizer the "
        "reasoning does not cover loses speed, never correctness.",
    )

    _h(doc, "3.2 Experimental setup", 2)
    _para(
        doc,
        "Both experiments use the held-out slice from §1.3, disjoint from the 500 MB the two matched-vocabulary "
        "tokenizers were trained on. Efficiency is total bytes divided by total tokens emitted, compared across "
        "30 tokenizers: the two trained here, the released 128k SuperBPE checkpoint, and 27 production tokenizers "
        "spanning 32k to 262k vocabulary.",
    )
    _para(
        doc,
        "For throughput, both engines load the same exported tokenizer.json and are handed the same pre-split "
        "documents, so the only variable is which engine does the work. HuggingFace gets its fastest available "
        "path, encode_batch_fast, with full parallelism, and the figure reported is the minimum of nine repeats. "
        "tiktoken does not appear at all: it cannot represent a SuperBPE tokenizer, so its absence is an "
        "exclusion rather than a slow result.",
    )
    _para(
        doc,
        "Those nine repeats bound variance within a single process, not between processes, and the distinction "
        "matters here: five runs at identical settings put the SuperBPE row at 618, 730, 745, 745 and 829 MB/s, a "
        "spread of about ±16%, against ±4% on the HuggingFace columns — 6.3, 6.5 and 6.8 MB/s across the same "
        "runs. The asymmetry is not a property of either engine but of the durations being compared: "
        "supergigatoken finishes the slice in **137 ms** where HuggingFace needs **15.9 s**, a measurement roughly "
        "**116× shorter**, so a fixed amount of OS scheduling noise across eight worker threads is a far larger "
        "fraction of it. The figures below come from one run taken with the machine otherwise idle, which landed "
        "within 0.5% of those five runs' mean.",
    )
    _para(
        doc,
        "A sixth run read 511 MB/s and is excluded, on evidence rather than on grounds of being inconvenient: "
        "three unrelated Python processes were started on the machine while it was in flight, and the damage "
        "distributes exactly as the argument above predicts. Contention moved the 137 ms SuperBPE row by **−31%** "
        "and the 25 s plain-BPE row by **−0.9%** — the same effect seen from the other side, since the long "
        "measurement amortises what the short one cannot. The practical consequence for reading this report is "
        "that the fast columns carry meaningful uncertainty in their last two digits while the ratios are stable, "
        "and that a throughput figure for an engine this fast is a statement about the machine's scheduler as much "
        "as about the code.",
    )
    _para(
        doc,
        "The released 128k checkpoint is measured on both engines. It ships an explicit Split-regex "
        "pretokenizer, which the loader now recognises as a bounded Superword scheme, so the two-level encoder "
        "runs on it rather than only on tokenizers trained here.",
    )

    _h(doc, "3.3 Results", 2)
    _para(doc, "Efficiency first, since it is the property that motivates the method at all:")
    _table(
        doc,
        "Encoding efficiency on the held-out slice; 7 of the 30 tokenizers measured. Higher bytes/token is better.",
        ["Tokenizer", "Vocab", "Bytes/token", "Tokens per 100 MB"],
        [
            ["Released SuperBPE (Liu et al.)", "128k", "6.23", "16.0M"],
            ["supergigatoken (SuperBPE, train_superbpe)", "50k", "5.67", "17.6M"],
            ["gpt-oss / Phi-4-mini", "200k", "4.70", "21.2M"],
            ["Llama 3.1", "128k", "4.67", "21.4M"],
            ["Gemma 4", "262k", "4.51", "22.1M"],
            ["gigatoken (plain BPE, train_bpe)", "50k", "4.49", "22.2M"],
            ["GPT-2", "50k", "4.41", "22.6M"],
        ],
        bold_col0=True,
    )
    _figure(
        doc,
        "superbpe_efficiency.png",
        "Left: bytes per token against vocabulary size across all 30 tokenizers measured — standard tokenizers "
        "trace a shallow trend under the whitespace ceiling, while SuperBPE sits above it. Right: the controlled "
        "matched-vocabulary comparison.",
    )
    _para(
        doc,
        "The controlled comparison is rows two and six: same corpus, same vocabulary size, same engine, differing "
        "only in whether stage 2 lifted the whitespace restriction. That is worth **20.7% fewer tokens** for the "
        "same text. The surrounding rows make a second point the controlled pair cannot — a 50k SuperBPE is more "
        "efficient than every standard tokenizer measured, including ones with four to five times the "
        "vocabulary, which is the ceiling from §1.1 showing up in practice. Then throughput, same slice:",
    )
    _table(
        doc,
        "Encoding throughput on the held-out slice. Both engines are given the same tokenizer.json and the same documents.",
        ["Tokenizer", "supergigatoken", "HuggingFace", "Speedup"],
        [
            ["SuperBPE 50k (Superword two-level encoder)", "730.0 MB/s", "6.3 MB/s", "116×"],
            ["Released SuperBPE 128k (superword_bounded outer scheme)", "310.6 MB/s", "4.9 MB/s", "63×"],
            ["plain BPE 50k (gigatoken's subword encoder)", "2297.1 MB/s", "4.0 MB/s", "577×"],
        ],
        bold_col0=True,
    )
    _figure(
        doc,
        "superbpe_throughput.png",
        "SuperBPE encoding throughput on the held-out slice — the same tokenizer.json and the same documents on both sides.",
        # A single two-bar comparison, so it does not want the full column the
        # multi-panel figures get. Narrowing it further to squeeze it in below
        # Table 8 was tried and abandoned: only 1.7" is left there, which would
        # mean a 1.8"-wide chart -- illegible, to save half a page.
        width_in=3.0,
    )
    _para(
        doc,
        "In tokens rather than bytes, the SuperBPE row is 128.8 against 1.11 million tokens per second. At corpus "
        "scale the difference stops being a benchmark: 1 TB tokenizes in roughly **0.4 hours** of single-machine "
        "CPU time at 730 MB/s, against roughly **44 hours** at HuggingFace's 6.3 MB/s — the difference between a "
        "step in a data pipeline and a scheduling problem.",
    )
    _para(
        doc,
        "The released checkpoint's row isolates what the two-level scheme itself contributes, because the same "
        "binary can encode it either way. Measured in-process over a 33.5 MB slice, min of five rounds, two-level encoding runs at "
        "**85.8 MB/s** against **31.6 MB/s** for the plain single-pretoken path on the identical tokenizer — "
        "**2.72×** — with the two token streams asserted identical across all 6,819 documents. That figure is lower "
        "than the table's 310.6 MB/s because it is a single-threaded in-Rust comparison rather than the parallel "
        "batch path; the ratio, not the level, is what it establishes.",
    )
    _para(
        doc,
        "Two-level encoding recovers most of the loss but not all of it. At 730 MB/s it remains about **3.1× "
        "below** gigatoken's plain subword path at the same vocabulary, and that remaining cost splits **33% level "
        "1 and 67% level 2** — so the merge, not the splitting, is what is left to attack. On the inherited engine, "
        "where whitespace pretokenization is available, gigatoken encodes GPT-2 at 24.5 GB/s on a 144-core EPYC; "
        "that matrix belongs upstream (benchmarks/compare/SUBWORD_THROUGHPUT.md) and is cited only to show that "
        "adding SuperBPE cost the subword path nothing.",
    )

    # ------------------------------------------------------------------
    _h(doc, "4. Limitations", 1)
    for item in [
        "Outcome parity, not merge identity. Stage-2 units are line-bounded, so the merge sequences are not "
        "byte-identical to the reference's. If bit-exact reproduction of the published tokenizer is a "
        "requirement, this does not deliver it.",
        "The released 128k checkpoint's own outer regex has no SIMD scanner: its boundaries are not a subset of "
        "the stage-1 ones, so the existing vectorised boundary harvest cannot be filtered down to it and the "
        "walker is scalar. At 578.6 MB/s that scan is 15% of the checkpoint's two-level cost, which caps a "
        "vectorised replacement at 1.18×, so it is deferred rather than done.",
        "SuperBPE encoding is roughly 3.1× below the plain subword path at the same vocabulary. The 116× figure "
        "is against HuggingFace on the same SuperBPE tokenizer, not against ordinary subword tokenization, and it "
        "carries the ±16% between-process spread described in §3.2.",
        "The +4.0 average and +8.2 MMLU results are Liu et al.'s, at a 200k vocabulary with a 180k transition "
        "point on an 8B model. They are evidence the method works, not a measurement of a 50k tokenizer trained "
        "here; only a pretraining run at your own scale would confirm the transfer.",
        "The transition point is a hyperparameter, not a constant to copy: the paper finds t=80k most efficient at a 200k vocabulary but t=180k best on downstream tasks.",
        "Stage-2 training is linear in unit length, so large vocabularies over hundreds of MB are minutes-scale, and memory grows with the unit set.",
        "All measurements come from a single Intel 8-core desktop CPU, so the ratios are not a claim about scaling "
        "across core counts or architectures. Windows is also lightly tested — prefer Linux or WSL for performance "
        "work, though the evaluation suite does run natively there.",
    ]:
        _para(doc, item, style="List Bullet")

    # ------------------------------------------------------------------
    _h(doc, "5. Conclusion", 1)
    _para(
        doc,
        "SuperBPE's two practical costs both turn out to be implementation costs rather than properties of the "
        "method. Training the tokenizer is **8.0×** faster than the only other implementation and succeeds at "
        "corpus sizes where that one runs out of memory, while learning a tokenizer that agrees with it on "
        "**47,742 of 50,000** vocabulary entries at **Spearman 0.999** on merge order. Encoding with it is "
        "**116×** faster than HuggingFace on the same tokenizer, with bit-identical output, because the trained "
        "merge table itself reveals which boundaries a merge can cross and therefore how to split the work into "
        "two levels. Neither result trades away quality: the tokenizer packs **20.7% more text** into each token "
        "than plain BPE trained identically.",
    )
    _para(
        doc,
        "What remains open is stated in §4, and the largest item is measured rather than guessed: encoding sits "
        "**3.1× below** the plain subword path, with **67%** of the remaining cost in the level-2 merge, which is "
        "where further work belongs. The claim this report does not make is the downstream one — that belongs to "
        "Liu et al., at a different vocabulary and model scale. What it does establish is that the method no longer "
        "costs hours to train or days to apply to a pretraining corpus, which is the precondition for testing that "
        "claim at one's own scale.",
    )

    # ------------------------------------------------------------------
    _h(doc, "Works cited", 1)
    for ref in [
        "Gage, P. (1994). A New Algorithm for Data Compression. The C Users Journal, 12(2), 23–38.",
        "Gokaslan, A., & Cohen, V. (2019). OpenWebText Corpus. skylion007.github.io/OpenWebTextCorpus",
        "Liu, A., Hayase, J., Hofmann, V., Oh, S., Smith, N. A., & Choi, Y. (2025). SuperBPE: Space Travel for "
        "Language Models. Second Conference on Language Modeling (COLM 2025). arXiv:2503.13423. Figures quoted "
        "here are from the v3 camera-ready.",
        "Moi, A., & Patry, N. (2023). HuggingFace Tokenizers. Software. github.com/huggingface/tokenizers",
        "OpenAI (2022). tiktoken. Software. github.com/openai/tiktoken",
        "Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI technical report.",
        "Rød, M. (2026). Gigatoken: SIMD and Cache Hierarchies for 1000× Faster Byte-Pair Encoding Tokenization on Modern CPUs. Software. github.com/marcelroed/gigatoken",
        "Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. Proceedings of ACL 2016, 1715–1725. arXiv:1508.07909.",
    ]:
        p = doc.add_paragraph()
        # Hanging indent: the author name starts at the margin and the rest of
        # the entry is inset, so the list scans by author. Entries are their own
        # visual unit, so they need less separation than body paragraphs.
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.space_after = Pt(2)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(ref)
        run.font.size = Pt(8.5)

    _note(
        doc,
        f"Measured figures: benchmarks/superbpe/REPORT.md and results_trainer.json — encoding throughput at commit "
        f"{MEASURED_COMMIT} ({MEASURED_DATE}), training, efficiency and parity at {TRAINED_COMMIT} and unchanged "
        f"since. OpenWebText, Intel 8-core, vocabulary 50000 / transition 40000. Emitted by "
        "efficiency.py, throughput.py, parity.py, vocab_diff.py and trainer_vs_hf.py, aggregated by report.py; "
        "figures by plot_readme.py. Subword throughput matrix: benchmarks/compare/SUBWORD_THROUGHPUT.md "
        "(upstream gigatoken). Prepared with AI assistance; every number is traceable to one of those sources.",
    )

    doc.save(path)
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = os.path.join(HERE, "supergigatoken-for-llm-training.docx")
    p.add_argument("--out", default=default)
    build(p.parse_args().out)


if __name__ == "__main__":
    main()
