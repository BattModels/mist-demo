import pytest
import json
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.cli import LightningCLI
from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts
import torch
from torch.utils.data import DataLoader


class MockedModel(LightningModule):
    def __init__(self, vocab_size):
        self.save_hyperparameters()
        super().__init__()

    def forward(self, input):
        return input

    def training_step(self, batch, batch_idx):
        self.forward(batch)  # Mock calling forward
        return torch.zeros(1, requires_grad=True)

    def configure_optimizers(self):
        pass


class MockedData(LightningDataModule):
    def __init__(self, tokenizer: str):
        self.tokenizer = tokenizer
        self.save_hyperparameters()
        super().__init__()

    def train_dataloader(self):
        return DataLoader(range(5), batch_size=1)


@pytest.fixture()
def cli(tmp_path):
    return LightningCLI(
        trainer_defaults={
            "max_steps": 2,
            "default_root_dir": tmp_path,
        },
        model_class=MockedModel,
        datamodule_class=MockedData,
        save_config_callback=SaveConfigWithCkpts,
        args=["fit", "--data.tokenizer=smirk", "--model.vocab_size=256"],
    )


def test_ckpt(cli):
    # Locate callback
    print(cli.trainer.callbacks)
    cb = list(
        filter(lambda cb: isinstance(cb, SaveConfigWithCkpts), cli.trainer.callbacks)
    )
    assert len(cb) == 1
    cb: SaveConfigWithCkpts = cb[0]

    assert cb.config_path is not None
    assert cb.config_path.is_dir()
    assert cb.config_path.joinpath("config.json").is_file()
    assert cb.config_path.joinpath("model_hparams.json").is_file()

    # Check that the dataloader config is saved
    data_config = {"tokenizer": "smirk"}
    assert dict(cb.config["data"]) == data_config
    with open(cb.config_path.joinpath("config.json"), "r") as fid:
        assert json.load(fid)["data"] == data_config

    # Check that the model config is saved
    with open(cb.config_path.joinpath("model_hparams.json"), "r") as fid:
        model_config = json.load(fid)
    assert model_config["class_path"] == __name__ + ".MockedModel"
    assert model_config["init_args"] == {"vocab_size": 256}
    assert "version" in model_config.keys()
