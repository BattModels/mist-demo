from smirk import SmirkTokenizerFast
from transformers import (AutoTokenizer,
                          PreTrainedTokenizerBase)
class DataSetupMixin:

    def setup_tokenizer(self, tokenizer: str):
        # Locate Tokeniser and dataset
        if tokenizer.startswith("smirk"):
            self.tokenizer = SmirkTokenizerFast(is_smiles=(tokenizer == "smirk"))
        else:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                tokenizer,
                trust_remote_code=True,
                cache_dir=".cache",  # Cache Tokenizer in working directory
            )