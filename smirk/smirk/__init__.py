# Import Rust Binding
from importlib.resources import files
from pathlib import Path
from typing import Union

from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils.generic import PaddingStrategy

from . import smirk as rs_smirk

VOCAB_FILE = str(Path(__file__).parent.parent.joinpath("vocab_smiles.json"))
# Expose chemically_consistent_split
from .smirk import chemically_consistent_split


class SmirkTokenizerFast(PreTrainedTokenizerBase):
    def __init__(self, **kwargs):
        # Create SmirkTokenizer
        default_vocab_file = str(files("smirk").joinpath("vocab_smiles.json"))
        if tokenizer_file := kwargs.pop("tokenizer_file", None):
            tokenizer = rs_smirk.SmirkTokenizer.from_file(tokenizer_file)
        elif vocab_file := kwargs.pop("vocab_file", default_vocab_file):
            # padding = kwargs.pop("padding", False)
            tokenizer = rs_smirk.SmirkTokenizer(vocab_file)
        self._tokenizer = tokenizer

        # Add special tokens
        kwargs.update(tokenizer.special_tokens)

        super().__init__(**kwargs)

    def __len__(self) -> int:
        """Size of the full vocab with added tokens"""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def __repr__(self):
        return self.__class__.__name__

    def is_fast(self):
        return True

    def batch_decode_plus(self, ids, **kwargs):
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)
        return self._tokenizer.decode_batch(ids)(
            ids, skip_special_tokens=skip_special_tokens
        )

    def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        add_special_tokens = kwargs.pop("add_special_tokens", True)
        encoding = self._tokenizer.encode_batch(
            batch_text_or_text_pairs, add_special_tokens=add_special_tokens
        )
        batch = BatchEncoding(
            data={k: [dic[k] for dic in encoding] for k in encoding[0]},
            encoding=encoding,
            n_sequences=len(encoding),
        )
        if kwargs.pop("padding_strategy") is not PaddingStrategy.DO_NOT_PAD:
            return pad_without_fast_tokenizer_warning(
                self, batch, return_tensors=kwargs.pop("return_tensors", "pt"), **kwargs
            )
        return batch

    def _encode_plus(
        self, text: str, add_special_tokens: bool = True, **kwargs
    ) -> BatchEncoding:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return BatchEncoding(data=encoding, encoding=encoding, n_sequences=1)

    def _decode(self, token_ids, **kwargs):
        skip_special_tokens = kwargs.get("skip_special_tokens", False)
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def get_vocab(self):
        return self._tokenizer.get_vocab(with_added_tokens=True)

    def convert_tokens_to_ids(
        self, tokens: Union[str, list[str]]
    ) -> Union[int, list[int]]:
        vocab = self.get_vocab()
        if isinstance(tokens, str):
            return vocab[tokens]
        return [vocab[token] for token in tokens]
