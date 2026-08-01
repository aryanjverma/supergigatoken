use crate::bpe::simple_bpe_merge;
use crate::token::TokenId;
use dashmap::DashMap;
use indicatif::ProgressBar;
use itertools::Itertools;
use priority_queue::PriorityQueue;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::collections::{BTreeSet, HashMap};

use std::hash::Hash;

#[derive(Clone)]
pub struct Word {
    pub symbols: Vec<u32>,
    pub word_count: isize,
}

type Pair = (u32, u32);

fn count_pairs(words: &[Word]) -> HashMap<Pair, isize> {
    let mut symbol_counts: HashMap<Pair, isize> = HashMap::new();
    for word in words.iter() {
        if word.symbols.len() < 2 {
            continue;
        }
        for i in 0..word.symbols.len() - 1 {
            let pair = (word.symbols[i], word.symbols[i + 1]);
            let count = symbol_counts.entry(pair).or_insert(0);
            *count += word.word_count;
        }
    }
    symbol_counts
}

fn update_word(
    w: &mut Word,
    pair: Pair,
    new_symbol: u32,
    mut record_changes: impl FnMut((u32, u32), isize),
) {
    let mut i = 0;
    while i < w.symbols.len() - 1 {
        if w.symbols[i] == pair.0 && w.symbols[i + 1] == pair.1 {
            // Perform the merge
            if i >= 1 {
                record_changes((w.symbols[i - 1], pair.0), -w.word_count);
                record_changes((w.symbols[i - 1], new_symbol), w.word_count);
            }
            if w.symbols.len() >= 3 && i <= w.symbols.len() - 3 {
                record_changes((pair.1, w.symbols[i + 2]), -w.word_count);
                record_changes((new_symbol, w.symbols[i + 2]), w.word_count);
            }
            w.symbols[i] = new_symbol;
            w.symbols.remove(i + 1);
        }
        i += 1;
    }
}

/// Unsafe hack to parallelize as efficiently as possible.
/// The borrow checker doesn't allow several mutable references to the same underlying array, so
/// we need to do unsafe dereferencing.
/// This is only safe if you there is _only one_ mutable reference to the underlying value.
#[derive(Clone)]
struct SendPtr(*mut Word);

unsafe impl Sync for SendPtr {}
unsafe impl Send for SendPtr {}

/// Update words by merging the given pair into a new symbol.
/// Update the contained_in_words map to _add_ associations between the newly created pairs and the words they are contained in (we don't remove old ones, though they will be stale).
/// Return a map of pair -> change in count (can be negative) to update the priority queue.
fn update_words(
    words: &mut [Word],
    contained_in_words: &mut HashMap<(u32, u32), BTreeSet<u32>>,
    pair: Pair,
    new_symbol: u32,
) -> DashMap<(u32, u32), isize, FxBuildHasher> {
    let count_changes: DashMap<(u32, u32), isize, FxBuildHasher> = DashMap::default();

    let n_threads = rayon::current_num_threads();

    // Iterate through all words containing first or second
    let word_idcs = &contained_in_words[&(pair.0, pair.1)];
    let words_ptr = SendPtr(words.as_mut_ptr());

    // TODO(perf): There is a lot of contention on this map early in merging, since the updated pairs overlap a lot in the beginning.
    // Pair -> Word, pair was added to the word, make sure to update contained_in_words
    let contained_updates: DashMap<(u32, u32), BTreeSet<u32>, FxBuildHasher> = DashMap::default();

    if word_idcs.len() > 2 * n_threads {
        word_idcs
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .par_chunks(word_idcs.len().div_ceil(n_threads))
            .for_each(|idcs_chunk| {
                for &i in idcs_chunk {
                    // Smuggle in a mutable reference to the word
                    let local_words_ptr = words_ptr.clone();
                    // SAFETY: Only this thread has access to this word, since word_idcs is a set of unique indices.
                    let word = unsafe { &mut *local_words_ptr.0.add(i as usize) };
                    let count_changes = |pair, change| {
                        if change > 0 {
                            // Was added to the word, need to track this immediately, since other threads might subtract
                            contained_updates.entry(pair).or_default().insert(i);
                        }
                        *count_changes.entry(pair).or_default() += change;
                    };
                    update_word(word, pair, new_symbol, count_changes);
                }
            });
    } else {
        // Single-threaded for small updates
        word_idcs.iter().copied().for_each(|i| {
            // Smuggle in a mutable reference to the word
            let local_words_ptr = words_ptr.clone();
            // SAFETY: Only this thread has access to this word, since word_idcs is a set of unique indices.
            let word = unsafe { &mut *local_words_ptr.0.add(i as usize) };
            let count_changes = |pair, change| {
                if change > 0 {
                    // Was added to the word, need to track this
                    contained_updates.entry(pair).or_default().insert(i);
                }
                *count_changes.entry(pair).or_default() += change;
            };
            update_word(word, pair, new_symbol, count_changes);
        });
    }

    for (pair, mut word_idcs) in contained_updates.into_iter() {
        let set = contained_in_words.entry(pair).or_default();
        set.append(&mut word_idcs);
    }

    count_changes
}

pub fn assemble_token(token: u32, symbols: &[Vec<u8>]) -> String {
    symbols[token as usize]
        .iter()
        .map(|x| *x as char)
        .collect::<String>()
}

pub struct BPEResult {
    pub vocab: HashMap<u32, Vec<u8>>,
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
}

/// How to break ties when multiple pairs have the same frequency count.
#[derive(Clone, Copy, Debug, Default)]
pub enum TieBreaking {
    /// Compare by token IDs remapped to match HuggingFace tokenizers' BpeTrainer
    /// initial vocabulary ordering (ByteLevel unicode codepoint order for bytes 0-255).
    #[default]
    HuggingFace,
    /// Compare by raw (u32, u32) token IDs (byte value = token ID).
    RawTokenIds,
    /// Assemble each token's bytes into a string and compare lexicographically.
    AssembledBytes,
}

/// Build a mapping from byte value (0-255) to the rank it would receive in
/// HuggingFace's BpeTrainer initial vocabulary. HF sorts the ByteLevel alphabet
/// by unicode codepoint: printable ASCII/Latin-1 bytes keep their codepoint,
/// while the remaining 68 bytes are remapped to U+0100..U+0143.
fn build_byte_to_hf_rank() -> [u32; 256] {
    let mut byte_to_cp = [0u32; 256];
    let mut n = 0u32;
    for b in 0..=255u8 {
        let is_allowed = matches!(b, 33..=126 | 161..=172 | 174..=255);
        if is_allowed {
            byte_to_cp[b as usize] = b as u32;
        } else {
            byte_to_cp[b as usize] = 256 + n;
            n += 1;
        }
    }

    // Sort bytes by their unicode codepoint, then record position as rank
    let mut bytes_sorted: Vec<u8> = (0..=255).collect();
    bytes_sorted.sort_by_key(|&b| byte_to_cp[b as usize]);

    let mut rank = [0u32; 256];
    for (i, &b) in bytes_sorted.iter().enumerate() {
        rank[b as usize] = i as u32;
    }
    rank
}

pub fn train_bpe<K: AsRef<[u8]> + Eq + Hash>(
    counts: HashMap<K, usize, FxBuildHasher>,
    vocab_size: usize,
    special_tokens: Vec<String>,
    tie_breaking: TieBreaking,
) -> BPEResult {
    // Indicates which word indices contain a given symbol
    let mut contained_in_words: HashMap<(u32, u32), BTreeSet<u32>> = HashMap::new();
    let mut contained_in_words_arr = vec![vec![vec![]; 256]; 256];
    let mut words: Vec<Word> = counts
        .into_iter()
        .enumerate()
        .map(|(word_i, (word, count))| {
            // At first we have only bytes, so we won't need to hash the u32 pairs
            let word_symbols: Vec<u32> = word.as_ref().iter().map(|&b| b as u32).collect();
            for c in word_symbols.iter().copied().tuple_windows::<(u32, u32)>() {
                contained_in_words_arr[c.0 as usize][c.1 as usize].push(word_i as u32);
            }
            Word {
                symbols: word_symbols,
                word_count: count as isize,
            }
        })
        .collect();

    for (i, j) in (0..256).cartesian_product(0..256) {
        if !contained_in_words_arr[i][j].is_empty() {
            contained_in_words.insert(
                (i as u32, j as u32),
                BTreeSet::from_iter(contained_in_words_arr[i][j].iter().copied()),
            );
        }
    }
    drop(contained_in_words_arr);

    println!("{} unique words", words.len());

    // Symbols 0 through 255 are the raw bytes; special tokens follow.
    let mut symbols: Vec<Vec<u8>> = (0..=255).map(|x| vec![x]).collect();
    symbols.extend(
        special_tokens
            .into_iter()
            .map(|x| x.bytes().collect::<Vec<u8>>()),
    );

    let merges = run_merges(
        &mut words,
        &mut contained_in_words,
        &mut symbols,
        vocab_size,
        tie_breaking,
    );

    let vocab: HashMap<_, _> = symbols
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v))
        .collect();

    BPEResult { vocab, merges }
}

/// The shared BPE merge loop, reused by both stage-1 (`train_bpe`) and
/// stage-2 (`train_superbpe_stage2`) training. Repeatedly merges the
/// highest-count symbol pair (ties broken per `tie_breaking`) into a new
/// symbol appended to `symbols`, until `symbols.len()` reaches
/// `max_symbols` or no pair remains. Returns the merges in priority order.
///
/// The starting state must be consistent: every adjacent pair occurring in
/// some word must key an entry in `contained_in_words`, and every symbol id
/// used by a word must index `symbols`. This holds for a byte-seeded start
/// (`train_bpe`) and for a seeded resume from an existing vocabulary
/// (`train_superbpe_stage2`).
fn run_merges(
    words: &mut [Word],
    contained_in_words: &mut HashMap<(u32, u32), BTreeSet<u32>>,
    symbols: &mut Vec<Vec<u8>>,
    max_symbols: usize,
    tie_breaking: TieBreaking,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    // Build HF rank table for tie-breaking (only used in HuggingFace mode)
    let hf_rank = build_byte_to_hf_rank();

    let symbol_counts = count_pairs(words);
    let mut pq = PriorityQueue::new();
    symbol_counts.into_iter().for_each(|(pair, count)| {
        pq.push(pair, count);
    });

    let mut merges = vec![];

    println!("Starting merges");
    let bar = ProgressBar::new(max_symbols as u64).with_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar}] {pos}/{len} ({eta})")
            .unwrap(),
    );

    while !pq.is_empty() && symbols.len() < max_symbols {
        bar.set_position(symbols.len() as u64);
        let pair = {
            let (first_pair, first_count) = pq.pop().unwrap();
            let mut tied_pairs = vec![first_pair];
            while let Some((_next_pair, &next_count)) = pq.peek() {
                if next_count != first_count {
                    break;
                }
                tied_pairs.push(pq.pop().unwrap().0);
            }
            // Find the smallest pair according to the chosen tie-breaking rule
            let mut smallest_pair = first_pair;
            match tie_breaking {
                TieBreaking::HuggingFace => {
                    // Remap initial byte token IDs to HF's ByteLevel unicode
                    // codepoint ordering before comparison.
                    let remap = |id: u32| -> u32 {
                        if id < 256 {
                            hf_rank[id as usize]
                        } else {
                            id
                        }
                    };
                    for &pair in &tied_pairs {
                        let remapped = (remap(pair.0), remap(pair.1));
                        let remapped_smallest =
                            (remap(smallest_pair.0), remap(smallest_pair.1));
                        if remapped < remapped_smallest {
                            smallest_pair = pair;
                        }
                    }
                }
                TieBreaking::RawTokenIds => {
                    for &pair in &tied_pairs {
                        if pair < smallest_pair {
                            smallest_pair = pair;
                        }
                    }
                }
                TieBreaking::AssembledBytes => {
                    let assemble_pair = |(p0, p1)| {
                        (assemble_token(p0, symbols), assemble_token(p1, symbols))
                    };
                    for pair in tied_pairs.iter().copied() {
                        if assemble_pair(pair) < assemble_pair(smallest_pair) {
                            smallest_pair = pair;
                        }
                    }
                }
            }

            for pair in tied_pairs {
                if pair != smallest_pair {
                    pq.push(pair, first_count);
                }
            }

            smallest_pair
        };

        // Merge the pair
        let new_symbol: Vec<u8> = [&symbols[pair.0 as usize], &symbols[pair.1 as usize]]
            .into_iter()
            .flatten()
            .copied()
            .collect();

        merges.push((
            symbols[pair.0 as usize].clone(),
            symbols[pair.1 as usize].clone(),
        ));

        symbols.push(new_symbol);

        let count_changes = update_words(
            words,
            contained_in_words,
            pair,
            symbols.len() as u32 - 1,
        );

        for (pair, change) in count_changes.into_iter() {
            let found_item = pq.change_priority_by(&pair, |p| *p += change);
            if !found_item {
                pq.push(pair, change);
            }
        }
    }
    bar.finish();

    merges
}

/// Build the pair -> word-index inverted index for an arbitrary set of
/// words whose symbols may be any token ids (not only bytes). The
/// byte-seeded stage-1 path in [`train_bpe`] builds this with a dense
/// 256x256 scratch table; this general version seeds a resumed
/// (stage-2) training run.
fn build_contained_in_words(words: &[Word]) -> HashMap<(u32, u32), BTreeSet<u32>> {
    let mut map: HashMap<(u32, u32), BTreeSet<u32>> = HashMap::new();
    for (word_i, word) in words.iter().enumerate() {
        for pair in word.symbols.iter().copied().tuple_windows::<(u32, u32)>() {
            map.entry(pair).or_default().insert(word_i as u32);
        }
    }
    map
}

/// Emit the `<= max_unit_len` byte units of a single line into `counts`,
/// splitting at UTF-8 character boundaries so a multi-byte character is
/// never cut in half.
fn push_line_units(
    line: &[u8],
    max_unit_len: usize,
    counts: &mut HashMap<Vec<u8>, usize, FxBuildHasher>,
) {
    let mut start = 0;
    while start < line.len() {
        let mut end = (start + max_unit_len).min(line.len());
        // Back off to a UTF-8 character boundary (a byte that is not a
        // 0b10xxxxxx continuation byte), unless that would make no progress.
        while end < line.len() && end > start && (line[end] & 0xC0) == 0x80 {
            end -= 1;
        }
        if end <= start {
            end = (start + max_unit_len).min(line.len());
        }
        *counts.entry(line[start..end].to_vec()).or_default() += 1;
        start = end;
    }
}

/// Relaxed ("superword") pretokenization for stage 2: whitespace splitting
/// is lifted, so a unit may span spaces, but units are still bounded on
/// document `separator`s and newlines and capped at `max_unit_len` bytes.
/// Bounding is deliberate -- the SuperBPE paper (section 2) notes that
/// unbounded stage-2 units are whole documents, which dedupe poorly and
/// blow up memory. Returns a map of unit bytes -> count.
fn relaxed_unit_counts(
    bytes: &[u8],
    separator: &[u8],
    max_unit_len: usize,
) -> HashMap<Vec<u8>, usize, FxBuildHasher> {
    let mut counts: HashMap<Vec<u8>, usize, FxBuildHasher> = HashMap::default();
    let max_unit_len = max_unit_len.max(1);

    let mut process_doc = |doc: &[u8], counts: &mut HashMap<Vec<u8>, usize, FxBuildHasher>| {
        for line in doc.split(|&b| b == b'\n') {
            if !line.is_empty() {
                push_line_units(line, max_unit_len, counts);
            }
        }
    };

    if separator.is_empty() {
        process_doc(bytes, &mut counts);
    } else {
        let mut last = 0;
        for pos in memchr::memmem::find_iter(bytes, separator) {
            process_doc(&bytes[last..pos], &mut counts);
            last = pos + separator.len();
        }
        process_doc(&bytes[last..], &mut counts);
    }

    counts
}

/// Stage 2 of SuperBPE: resume BPE training from a stage-1 result with
/// whitespace pretokenization lifted, learning superword merges.
///
/// The stage-1 vocabulary becomes the initial symbol table, and each
/// relaxed (superword) unit is seeded by encoding it with the stage-1
/// tokenizer via [`simple_bpe_merge`] (equivalent to encoding each
/// whitespace-delimited pretoken and concatenating, since no stage-1 merge
/// bridges whitespace). Merging then continues to `target_vocab_size`.
///
/// Returns the combined vocabulary and the stage-1 merges followed by the
/// stage-2 (superword) merges, in priority order.
pub fn train_superbpe_stage2(
    corpus: &[u8],
    separator: &[u8],
    stage1: BPEResult,
    target_vocab_size: usize,
    tie_breaking: TieBreaking,
    max_unit_len: usize,
) -> BPEResult {
    // Rebuild the contiguous id -> bytes symbol table from stage 1.
    let n_symbols = stage1.vocab.len();
    let mut symbols: Vec<Vec<u8>> = vec![Vec::new(); n_symbols];
    for (id, token_bytes) in stage1.vocab.into_iter() {
        symbols[id as usize] = token_bytes;
    }

    // bytes -> id, then a (TokenId, TokenId) -> TokenId merge map so we can
    // encode stage-2 units with the stage-1 tokenizer. Merged ids are
    // monotonic in merge order, so `simple_bpe_merge`'s "lowest merged id
    // first" priority reproduces stage-1's merge order.
    let mut bytes_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(n_symbols);
    for (id, token_bytes) in symbols.iter().enumerate() {
        bytes_to_id.entry(token_bytes.clone()).or_insert(id as u32);
    }

    let mut merge_map: HashMap<(TokenId, TokenId), TokenId, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(stage1.merges.len(), FxBuildHasher);
    for (left, right) in stage1.merges.iter() {
        let mut merged = left.clone();
        merged.extend_from_slice(right);
        let (Some(&li), Some(&ri), Some(&mi)) = (
            bytes_to_id.get(left),
            bytes_to_id.get(right),
            bytes_to_id.get(&merged),
        ) else {
            continue;
        };
        merge_map.insert((TokenId(li), TokenId(ri)), TokenId(mi));
    }
    drop(bytes_to_id);

    // Relaxed (superword) pretokenization, then seed each unit's symbols by
    // encoding it with the stage-1 tokenizer.
    let unit_counts = relaxed_unit_counts(corpus, separator, max_unit_len);
    println!("{} unique superword units", unit_counts.len());

    let mut words: Vec<Word> = Vec::with_capacity(unit_counts.len());
    for (unit, count) in unit_counts.into_iter() {
        let encoded = simple_bpe_merge(&merge_map, &unit);
        if encoded.is_empty() {
            continue;
        }
        words.push(Word {
            symbols: encoded.into_iter().map(|t| t.0).collect(),
            word_count: count as isize,
        });
    }

    let mut contained_in_words = build_contained_in_words(&words);

    let stage2_merges = run_merges(
        &mut words,
        &mut contained_in_words,
        &mut symbols,
        target_vocab_size,
        tie_breaking,
    );

    let mut merges = stage1.merges;
    merges.extend(stage2_merges);

    let vocab: HashMap<_, _> = symbols
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v))
        .collect();

    BPEResult { vocab, merges }
}
