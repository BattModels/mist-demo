from transformers import PreTrainedTokenizerBase


def load_tokenizer(name, **kwargs) -> PreTrainedTokenizerBase:
    # Locate Tokeniser and dataset
    unk_name = RuntimeError(f"Unknown tokenizer: {name}")
    if name == "SmilesPE/SPE_ChEMBL":
        from ..tokenize.spe import pretrained_spe_tokenizer

        return pretrained_spe_tokenizer()

    else:
        # Fall back to a HuggingFace Tokenizer
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            cache_dir=".cache",  # Cache Tokenizer in working directory
            **kwargs,
        )