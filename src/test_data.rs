//! Test-only lookup of the optional plain-text corpora under `~/data`.
//!
//! The Hub-cached counterpart is `test_hub`; these are the raw dumps the
//! differential tests read (OpenWebText, TinyStories), which are gigabytes and
//! are never committed or downloaded by a test.
//!
//! A test whose corpus is missing has to be able to say so and stop. A hard
//! `unwrap()` reports an absent optional download as a red test, which leaves
//! the suite unable to distinguish "this machine has no corpus" from "the
//! pretokenizer is wrong" — and that distinction is the only thing the suite is
//! for when a real bug has to be isolated. Skipping is conditional on absence
//! only: when the file is there the differential runs in full, and any error
//! *reading* it still panics.

use std::path::PathBuf;

/// `~/data/<rel>` when that file exists, else None.
pub(crate) fn corpus(rel: &str) -> Option<PathBuf> {
    let path = std::env::home_dir()?.join("data").join(rel);
    path.is_file().then_some(path)
}

/// [`corpus`], naming the path it could not find.
///
/// Use as the head of a test that cannot run without the corpus:
///
/// ```ignore
/// let Some(path) = corpus_or_skip("TinyStoriesV2-GPT4-valid.txt") else { return };
/// ```
///
/// The stderr line (shown by `cargo test -- --nocapture`, and always shown for
/// a test that then fails for another reason) is the record that the body was
/// skipped rather than checked, so a green run is never mistaken for coverage.
pub(crate) fn corpus_or_skip(rel: &str) -> Option<PathBuf> {
    let found = corpus(rel);
    if found.is_none() {
        eprintln!(
            "SKIP: ~/data/{rel} is absent. This corpus is not committed and no test downloads \
             it; place it there to run this differential."
        );
    }
    found
}

/// First `max_bytes` of `~/data/owt_train.txt`, truncated to a UTF-8 boundary,
/// or None when the corpus is absent.
///
/// Streamed rather than read whole: the file is ~12 GB and every caller wants a
/// prefix. Nine test modules had grown their own byte-identical copy of this,
/// each `expect`ing the open — which is why a CI runner (no `~/data`) failed 11
/// differentials at once with "No such file or directory" while this machine,
/// which happens to have the corpus, passed. Absence is the only condition that
/// skips: a read error on a corpus that *does* exist still panics.
pub(crate) fn owt_prefix_or_skip(max_bytes: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    let path = corpus_or_skip("owt_train.txt")?;
    let f = std::fs::File::open(&path).expect("opening a corpus that is present");
    let mut buf = Vec::with_capacity(max_bytes.min(1 << 30));
    f.take(max_bytes as u64).read_to_end(&mut buf).expect("reading owt_train.txt");
    // Same trim as the copies this replaces: drop the trailing partial
    // character so the slice is valid UTF-8, at most three pops.
    while !buf.is_empty() && std::str::from_utf8(&buf).is_err() {
        buf.pop();
    }
    Some(buf)
}
