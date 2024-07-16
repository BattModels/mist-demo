import importlib
import json
import os
from pathlib import Path

from jsonargparse import Namespace
from pytorch_lightning import Callback, LightningModule, Trainer


class SaveConfigWithCkpts(Callback):
    """Save Configuration with the model's checkpoints

    # Versions: Use Semantic Versioning for Model Checkpoint format

    ## Unnamed:
        - Stored `trainer.lightning_model.hparams` in model_hparms.json

    ## 0.2.0
        - Added version field to model_hparams.json
        - Added "class_path" field
        - Moved model hparams to "init_args" field

    ## 0.2.1
        - Save `JOB_CONFIG` to `job_config.json`
    """

    VERSION = "0.2.1"

    def __init__(
        self,
        config: Namespace,
        overwrite: bool = True,
    ) -> None:
        self.config = config
        self.overwrite = overwrite
        self.already_saved = False
        self.config_path = None

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