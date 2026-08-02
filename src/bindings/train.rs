//! Python binding for BPE training: the train_bpe function and the
//! conversion of its result into Python vocab/merges objects.

use super::sources::FileSource;
use crate::bpe_train;
use crate::input::file_source::FileSourceSpec;
use crate::input::{MmappedFile, Resource};
use crate::pretokenize;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
use std::path::PathBuf;

/// Vocab dict plus ordered merge pairs, as Python objects.
type PyVocabAndMerges<'py> = (
    Bound<'py, PyDict>,
    Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>,
);

fn bpe_result_to_python<'py>(
    py: Python<'py>,
    result: bpe_train::BPEResult,
) -> PyResult<PyVocabAndMerges<'py>> {
    let vocab_py = result
        .vocab
        .into_iter()
        .map(|(k, v)| (k, PyBytes::new(py, &v)))
        .sorted_by(|e1, e2| Ord::cmp(&e1.0, &e2.0))
        .into_py_dict(py);
    let merges_py: Vec<_> = result
        .merges
        .into_iter()
        .map(|(k, v)| (PyBytes::new(py, &k), PyBytes::new(py, &v)))
        .collect();
    Ok((vocab_py?, merges_py))
}

/// Reject a non-GPT-2 `pretokenizer` on an input path that cannot honor it.
///
/// The multi-file / parquet paths count pretokens inside `FileSourceSpec`,
/// which is not scheme-parameterized. Erroring is deliberate: silently
/// pretokenizing with GPT-2 after the caller asked for another scheme would
/// train a tokenizer that disagrees with its own declared pretokenizer.
fn require_default_pretokenizer(
    scheme: pretokenize::PretokenizerType,
    what: &str,
) -> PyResult<()> {
    if scheme != pretokenize::PretokenizerType::GPT2 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            format!(
                "pretokenizer must be 'gpt2' for {what}; pass bytes or a single \
                 text file path to train with another scheme"
            ),
        ));
    }
    Ok(())
}

fn parse_tie_breaking(s: &str) -> PyResult<bpe_train::TieBreaking> {
    match s {
        "huggingface" => Ok(bpe_train::TieBreaking::HuggingFace),
        "raw_token_ids" => Ok(bpe_train::TieBreaking::RawTokenIds),
        "assembled_bytes" => Ok(bpe_train::TieBreaking::AssembledBytes),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "tie_breaking must be 'huggingface', 'raw_token_ids', or 'assembled_bytes', got {other:?}"
        ))),
    }
}

#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (in_data, vocab_size, special_tokens, tie_breaking = "huggingface", separator = None, pretokenizer = "gpt2"))]
pub(crate) fn train_bpe<'py>(
    py: Python<'py>,
    in_data: Bound<'py, PyAny>,
    vocab_size: usize,
    special_tokens: Vec<String>,
    tie_breaking: &str,
    separator: Option<&[u8]>,
    pretokenizer: &str,
) -> PyResult<(
    Bound<'py, PyDict>,
    Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>,
)> {
    assert!(
        vocab_size <= 2_usize.pow(32),
        "vocab_size must be less than 2^32"
    );
    let tie_breaking = parse_tie_breaking(tie_breaking)?;
    let scheme = super::pretokenize::pretokenizer_scheme(pretokenizer)?;
    let separator = separator.unwrap_or(pretokenize::DEFAULT_SEPARATOR);

    // --- FileSource: multi-file parallel processing ---
    if let Ok(file_source) = in_data.extract::<FileSource>() {
        require_default_pretokenizer(scheme, "FileSource")?;
        let spec = FileSourceSpec {
            paths: file_source.paths,
            format: file_source.format,
        };
        let counts = spec.pretokenize().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "FileSource processing failed: {}",
                e
            ))
        })?;
        let result = bpe_train::train_bpe(counts, vocab_size, special_tokens, tie_breaking);
        return bpe_result_to_python(py, result);
    }

    // --- Single bytes or file path ---
    let mmap_resource;
    let bytes: &[u8] = if in_data.is_instance_of::<PyBytes>() {
        in_data.extract::<&[u8]>()?
    } else if let Ok(path) = in_data.extract::<PathBuf>() {
        if let Some(ext) = path.extension()
            && ext == "parquet"
        {
            require_default_pretokenizer(scheme, "parquet input")?;
            // A bare path takes the default column "text", matching
            // detect_default_format; use ParquetFileSource to choose another
            // column.
            let spec = FileSourceSpec {
                paths: vec![path],
                format: crate::input::file_source::DocFormat::Parquet {
                    column: "text".to_string(),
                },
            };
            let counts = spec
                .pretokenize()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            let result = bpe_train::train_bpe(counts, vocab_size, special_tokens, tie_breaking);
            return bpe_result_to_python(py, result);
        }
        mmap_resource = MmappedFile::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to open file {:?}: {}",
                path, e
            ))
        })?;
        mmap_resource.as_bytes()
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "in_data must be bytes, a path, or a FileSource",
        ));
    };

    let counts = pretokenize::pretokenize_par_bytes(bytes, separator, scheme);
    let result = bpe_train::train_bpe(counts, vocab_size, special_tokens, tie_breaking);
    bpe_result_to_python(py, result)
}

/// Train a SuperBPE tokenizer: two-stage BPE where stage 1 is ordinary
/// whitespace-pretokenized BPE up to `transition_point`, and stage 2
/// resumes with whitespace pretokenization lifted so merges can bridge
/// whitespace, learning "superword" tokens up to `vocab_size`.
///
/// Because the raw corpus is needed twice (stage 1 pretokenized, stage 2
/// relaxed), this currently accepts only in-memory bytes or a single text
/// file path -- not FileSource or parquet.
///
/// `pretokenizer` selects the stage-1 scheme (see
/// `PretokenizerType::NAMES`); stage 2 is always the relaxed superword
/// unit scheme. Pass `"superbpe_stage1"` to match the original SuperBPE
/// trainer's stage-1 regex, whose letter classes include `\p{M}` so
/// combining marks stay inside their letter run. The `"gpt2"` default
/// excludes `\p{M}`, which fragments scripts that write vowels as marks
/// (e.g. Devanagari) and — because stage 2 lifts all splitting — lets
/// stage 2 repair damage stage 1 caused, inflating the apparent superword
/// gain for exactly those scripts.
#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (in_data, vocab_size, transition_point, special_tokens, tie_breaking = "huggingface", separator = None, max_unit_len = 128, pretokenizer = "gpt2"))]
pub(crate) fn train_superbpe<'py>(
    py: Python<'py>,
    in_data: Bound<'py, PyAny>,
    vocab_size: usize,
    transition_point: usize,
    special_tokens: Vec<String>,
    tie_breaking: &str,
    separator: Option<&[u8]>,
    max_unit_len: usize,
    pretokenizer: &str,
) -> PyResult<(
    Bound<'py, PyDict>,
    Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>,
)> {
    assert!(
        vocab_size <= 2_usize.pow(32),
        "vocab_size must be less than 2^32"
    );
    if transition_point > vocab_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "transition_point ({transition_point}) must be <= vocab_size ({vocab_size})"
        )));
    }
    let tie_breaking = parse_tie_breaking(tie_breaking)?;
    let scheme = super::pretokenize::pretokenizer_scheme(pretokenizer)?;
    let separator = separator.unwrap_or(pretokenize::DEFAULT_SEPARATOR);

    // train_superbpe reads the corpus twice, so it supports only in-memory
    // bytes or a single (non-parquet) text file path for now.
    let mmap_resource;
    let bytes: &[u8] = if in_data.is_instance_of::<PyBytes>() {
        in_data.extract::<&[u8]>()?
    } else if let Ok(path) = in_data.extract::<PathBuf>() {
        if path.extension().is_some_and(|ext| ext == "parquet") {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "train_superbpe does not support parquet input; pass bytes or a text file path",
            ));
        }
        mmap_resource = MmappedFile::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to open file {:?}: {}",
                path, e
            ))
        })?;
        mmap_resource.as_bytes()
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "train_superbpe in_data must be bytes or a text file path",
        ));
    };

    // Stage 1: standard whitespace-pretokenized BPE up to the transition point.
    let counts = pretokenize::pretokenize_par_bytes(bytes, separator, scheme);
    let stage1 = bpe_train::train_bpe(counts, transition_point, special_tokens, tie_breaking);

    // Stage 2: resume with whitespace pretokenization lifted (superwords).
    let result = bpe_train::train_superbpe_stage2(
        bytes,
        separator,
        stage1,
        vocab_size,
        tie_breaking,
        max_unit_len,
    );

    bpe_result_to_python(py, result)
}
