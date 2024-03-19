from train import cli_main
from tempfile import TemporaryDirectory


def test_default():

    with TemporaryDirectory() as fake_data_dir:
        cli = cli_main(
            [
                "--data.path",
                fake_data_dir,
                "--data.tokenizer",
                "ibm/MoLFormer-XL-both-10pct",
            ]
        )
    assert cli.datamodule.vocab_size == cli.model.vocab_size
