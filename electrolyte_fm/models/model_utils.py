import json
from pathlib import Path

import pytorch_lightning as pl
from deepspeed.utils.zero_to_fp32 import \
    get_fp32_state_dict_from_zero_checkpoint


class DeepSpeedMixin:

    @classmethod
    def load(cls, checkpoint_dir, config_path=None):
        """Restore from a deepspeed checkpoint, mainly used for downstream tasks"""
        checkpoint_dir = Path(checkpoint_dir).resolve()
        config_path = config_path or checkpoint_dir.parent.parent.joinpath(
            "model_hparams.json"
        )
        assert (
            checkpoint_dir.is_dir()
        ), f"Missing deepspeed checkpoint director {checkpoint_dir}"
        assert config_path.is_file(), f"Missing model config file {config_path}"

        # Restore mode from config
        with open(config_path, "r") as fid:
            model_config = json.load(fid)
        model = cls(**model_config)

        # Load model weights from checkpoint
        state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
        model.load_state_dict(state, strict=True, assign=True)
        return model
    
    @staticmethod
    def get_encoder(model):
        raise NotImplementedError
        
    @classmethod
    def load_encoder(cls, checkpoint_dir, config_path=None):
        model = cls.load(checkpoint_dir, config_path)
        return cls.get_encoder(model=model)

class LoggingMixin(pl.LightningModule):
    
    def on_train_epoch_start(self) -> None:
        # Update the dataset's internal epoch counter
        self.trainer.train_dataloader.dataset.set_epoch(self.trainer.current_epoch)
        self.log(
            "train/dataloader_epoch",
            self.trainer.train_dataloader.dataset._epoch,
            rank_zero_only=True,
            sync_dist=True,
        )
        return super().on_train_epoch_start()

