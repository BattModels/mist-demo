from tempfile import TemporaryDirectory

from train import cli_main


def test_default():

    with TemporaryDirectory() as fake_data_dir:
        cli = cli_main(
            [
                "--data",
                "RobertaDataSet",
                "--data.path",
                fake_data_dir,
                "--data.tokenizer",
                "ibm/MoLFormer-XL-both-10pct",
                "--model",
                "RoBERTa",
                "--trainer.devices=1",
            ]
        )
    assert cli.datamodule.vocab_size == cli.model.vocab_size
