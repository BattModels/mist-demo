from tempfile import TemporaryDirectory
from random import randint


from pytorch_lightning.loggers import WandbLogger

from train import cli_main


def test_default():
    tag = f"testing-{randint(0, 128)}"
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
                f"--tags=['{tag}']",
            ]
        )
    assert cli.datamodule.vocab_size == cli.model.vocab_size
    assert isinstance(cli.trainer.logger, WandbLogger)
    assert cli.trainer.logger.experiment.tags == (tag,)
