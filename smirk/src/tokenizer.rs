use pyo3::{pyclass, pymethods, PyResult, FromPyObject, Python};
use pyo3::types::{PyAny, PyString};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::decoders::strip::Strip as DecodeStrip;
use tokenizers::normalizers::Strip;
use tokenizers::{EncodeInput, Encoding, InputSequence, PostProcessorWrapper, Result, TokenizerBuilder, TokenizerImpl};
use crate::pretokenizer::SmirkPreTokenizer;

#[pyclass]
pub struct SmirkTokenizer {
    tokenizer:TokenizerImpl<WordLevel, Strip, SmirkPreTokenizer, PostProcessorWrapper, DecodeStrip>
}


#[pymethods]
impl SmirkTokenizer {
    #[new]
    fn new(file: &str) -> Self {
        let model = WordLevel::from_file(file, "[UNK]".into()).unwrap();
        let tokenizer = TokenizerBuilder::new()
            .with_model(model)
            .with_pre_tokenizer(Some(SmirkPreTokenizer))
            .with_normalizer(Some(Strip::new(true, true)))
            .with_decoder(None::<DecodeStrip>)
            .with_post_processor(None::<PostProcessorWrapper>)
            .build()
            .unwrap();
        return Self { tokenizer }
    }

    #[pyo3(signature = (smile, add_special_tokens = true))]
    fn encode(&self, smile: &PyString, add_special_tokens: bool) -> PyResult<PyEncoding>{
        let input = EncodeInput::from(smile.to_str().unwrap());
        let encoding = self.tokenizer.encode_char_offsets(input, add_special_tokens).unwrap();
        return Ok(PyEncoding::from(encoding))
    }

    #[pyo3(signature = (examples, add_special_tokens = true))]
    fn encode_batch(&self, py: Python<'_>, examples: Vec<&PyString>, add_special_tokens: bool) -> PyResult<Vec<PyEncoding>>{
        let inputs: Vec<EncodeInput> = examples
            .into_iter()
            .map(|x| EncodeInput::from(x.to_str().unwrap()))
            .collect()
        ;
        // Release the GIL while tokenizing batch
        let out = py.allow_threads(|| {
            self.tokenizer
                .encode_batch_char_offsets(inputs, add_special_tokens)
                .unwrap()
                .into_iter()
                .map(|e| PyEncoding::from(e))
                .collect()
        });
        Ok(out)
    }
}

#[pyclass]
pub struct PyEncoding {
    #[pyo3(get, set)]
    ids: Vec<u32>,
    #[pyo3(get, set)]
    type_ids: Vec<u32>,
    #[pyo3(get, set)]
    attention_mask: Vec<u32>,
}

impl From<Encoding> for PyEncoding {
    fn from(encoding: Encoding) -> Self {
        Self {
            ids: encoding.get_ids().to_vec(),
            type_ids: encoding.get_type_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
        }
    }
}
