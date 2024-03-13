mod split;
mod pretokenizer;
mod tokenizer;

use pyo3::{prelude::*, types::PyString};
use split::split_chemically_consistent;
use tokenizer::SmirkTokenizer;

#[pyfunction]
fn chemically_consistent_split(a: &PyString) -> PyResult<Vec<String>> {
    Ok(split_chemically_consistent(a.to_str().unwrap()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn smirk(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chemically_consistent_split, m)?)?;
    m.add_class::<SmirkTokenizer>()?;
    Ok(())
}
