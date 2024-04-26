import smirk
from pathlib import Path
from transformers import PreTrainedTokenizerBase
from test_tokenize_smiles import smile_strings
from tokenizers.trainers import WordPieceTrainer


def test_smoke(smile_strings):
    tokenizer = smirk.SmirkTokenizerFast.chem_piece()
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    assert tokenizer.unk_token == "[UNK]"
    emb = tokenizer(smile_strings)
    out = [tokenizer.decode(x) for x in emb["input_ids"]]
    assert out == smile_strings


def test_train(smile_strings):
    tokenizer = smirk.SmirkTokenizerFast.chem_piece()
    tokenizer(smile_strings)
    vocab = Path(__file__).parent.parent.parent.joinpath(
        "electrolyte_fm",
        "pretrained_tokenizers",
        "zinc250k_vocab.txt",
    )
    assert vocab.is_file()
    tokenizer.train(files=[str(vocab)])
    tokenizer.add_special_tokens(smirk.SPECIAL_TOKENS)
    print(tokenizer.to_str())
    print(tokenizer(smile_strings))
    assert False
