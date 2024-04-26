mod pre_tokenizers;
mod gpe;
mod tokenizer;

use pyo3::prelude::*;
use tokenizer::SmirkTokenizer;

/// A Python module implemented in Rust.
#[pymodule]
fn smirk(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SmirkTokenizer>()?;
    Ok(())
}
