import smirk
from pathlib import Path
from test_tokenize_smiles import smile_strings
from test_fast_tokenizer import check_save


SMILE_TEST_FILE = Path(__file__).parent.joinpath("smiles.txt")


def test_train(smile_strings):
    tokenizer = smirk.SmirkTokenizerFast()
    trained = tokenizer.train([str(SMILE_TEST_FILE)])
    print(trained.to_str())
    check_save(trained)

    # Check inversion
    code = trained(smile_strings)
    decode = trained.batch_decode(code["input_ids"])
    assert decode == smile_strings
    assert trained.vocab_size > tokenizer.vocab_size
    print(trained.get_vocab())
    assert "[PAD]" not in trained._tokenizer.get_vocab(with_added_tokens=False)
    assert "[PAD]" in trained.get_vocab()


def test_vocab_size():
    tokenizer = smirk.SmirkTokenizerFast()
    trained = tokenizer.train([str(SMILE_TEST_FILE)], vocab_size=200)
    assert trained.vocab_size == 200


def test_multi_file():
    tokenizer = smirk.SmirkTokenizerFast()
    trained = tokenizer.train([str(SMILE_TEST_FILE)], vocab_size=200)
    assert trained.vocab_size == 200
