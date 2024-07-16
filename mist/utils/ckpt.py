import importlib
import json
import os
from pathlib import Path

from jsonargparse import Namespace
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.cli import LightningArgumentParser


class SaveConfigWithCkpts(Callback):
    """Save Configuration with the model's checkpoints

    # Versions: Use Semantic Versioning for Model Checkpoint format

    ## Unnamed:
        - Stored `trainer.lightning_model.hparams` in model_hparms.json

    ## 0.2.0
        - Added version field to model_hparams.json
        - Added "class_path" field
        - Moved model hparams to "init_args" field
    """

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        overwrite: bool = True,
    ) -> None:
        self.parser = parser
        self.config = config
        self.overwrite = overwrite
        self.already_saved = False
        self.config_path = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        log_dir = trainer.log_dir or Path.cwd()
        if logger := trainer.logger:
            config_path = Path(log_dir, str(logger.name), str(logger.version))
        else:
            config_path = Path(log_dir)
        self.config_path = config_path

        if trainer.is_global_zero:
            config_path.mkdir(parents=True, exist_ok=True)
            config_json = self.parser.dump(
                self.config,
                skip_none=False,
                skip_check=True,
                skip_link_targets=False,
                format="json",
            )
            with open(Path(config_path, "config.json"), "w") as config_file:
                config_file.write(config_json)

            # Save model hyperparameters
            with open(Path(config_path, "model_hparams.json"), "w") as fid:
                model_cls = trainer.lightning_module.__class__
                model_config = {
                    "version": "0.2.0",
                    "class_path": f"{model_cls.__module__}.{model_cls.__name__}",
                    "init_args": trainer.lightning_module.hparams,
                }
                json.dump(model_config, fid, default=lambda x: str(type(x)))

            # Save Environment
            with open(Path(config_path, "env.json"), "w") as fid:
                json.dump(dict(os.environ), fid, sort_keys=True)

            if logger := trainer.logger:
                logger.log_hyperparams({"cli": self.config.as_dict()})

    @staticmethod
    def load(checkpoint_dir: str | Path, config_path=None) -> LightningModule:
        """Restore from a deepspeed checkpoint, mainly used for downstream tasks"""
        checkpoint_dir = Path(checkpoint_dir).resolve()
        config_path = config_path or checkpoint_dir.parent.parent.joinpath(
            "model_hparams.json"
        )
        assert (
            checkpoint_dir.is_dir()
        ), f"Missing deepspeed checkpoint director {checkpoint_dir}"
        assert config_path.is_file(), f"Missing model config file {config_path}"

        with open(config_path, "r") as fid:
            config = json.load(fid)

        # Get model class name and config
        if "version" in config:
            cls_name = config["class_path"]
            model_config = config["init_args"]
        else:
            cls_name = "electrolyte_fm.models.roberta_base.RoBERTa"
            model_config = config

        # Import the model class and initialize the model
        import_path = cls_name.split(".")
        model_cls = importlib.import_module(
            ".".join(import_path[:-2])
        ).__getattribute__(import_path[-1])
        assert cls_name == f"{model_cls.__module__}.{model_cls.__name__}"
        model = model_cls(**model_config)
        model.configure_model()
        # Load model weights from the checkpoint
        from deepspeed.utils.zero_to_fp32 import (
            get_fp32_state_dict_from_zero_checkpoint,
        )

        state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
        model.load_state_dict(state, strict=True, assign=True)
        return model

    @staticmethod
    def get_ckpt_tokenizer(path: str | Path):
        print(f"loading tokenizer from {path}")
        path = Path(path)
        config_path = path.parent.parent.joinpath("config.json")
        assert config_path.is_file()
        with open(config_path, "r") as fid:
            config = json.load(fid)
        try:
            tokenizer = config["data"]["tokenizer"]
        except:
            tokenizer = config["data"]["init_args"]["tokenizer"]
        print(f"tokenizer: {tokenizer}")
        return tokenizer