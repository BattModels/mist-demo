from transformers import PreTrainedTokenizerBase


def load_tokenizer(name, **kwargs) -> PreTrainedTokenizerBase:
    # Locate Tokeniser and dataset
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=True,
        cache_dir=".cache",  # Cache Tokenizer in working directory
        **kwargs,
    )
