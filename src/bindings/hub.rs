//! pyo3 forwards for `load_tokenizer::hub` — the HuggingFace Hub
//! cache-or-download used by `gigatoken._load.hub`.

use crate::load_tokenizer::hub;
use pyo3::exceptions::{PyFileNotFoundError, PyPermissionError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

/// Path of `filename` from Hub repo `repo_id` at `revision`, served from
/// the standard HF cache, downloading into it first when absent.
#[pyfunction]
#[pyo3(signature = (repo_id, filename = "tokenizer.json", *, repo_type = "model", revision = "main"))]
pub fn hub_file(
    py: Python<'_>,
    repo_id: &str,
    filename: &str,
    repo_type: &str,
    revision: &str,
) -> PyResult<PathBuf> {
    let repo_type: hub::RepoType =
        repo_type.parse().map_err(|err: eyre::Report| PyValueError::new_err(err.to_string()))?;
    py.detach(|| hub::hub_file_in(repo_type, repo_id, filename, revision)).map_err(|err| {
        // Definite HTTP causes surface as the matching Python exception.
        match err.downcast_ref::<hub::FetchError>() {
            Some(fetch @ hub::FetchError::NotFound { .. }) => PyFileNotFoundError::new_err(fetch.to_string()),
            Some(fetch @ hub::FetchError::Unauthorized { .. }) => PyPermissionError::new_err(fetch.to_string()),
            // `{err:#}` rather than the default conversion: an eyre report
            // Displays only its outermost context, so a failed fetch reached
            // Python as "downloading tokenizer.json from Hub repo X" with the
            // actual reason — which URL, which status, which IO error — dropped.
            // That is the whole diagnostic value of the chain, and its absence
            // made an intermittent CI failure in
            // `test_tokenizer_from_repo_id_downloads_into_cache` un-triageable.
            None => PyRuntimeError::new_err(format!("{err:#}")),
        }
    })
}

/// Whether `name` is shaped like a HuggingFace Hub repo id.
#[pyfunction]
pub fn looks_like_repo_id(name: &str) -> bool {
    hub::looks_like_repo_id(name)
}

/// The discovered HuggingFace access token, or None.
#[pyfunction]
pub fn get_hf_token() -> Option<String> {
    hub::get_hf_token()
}
