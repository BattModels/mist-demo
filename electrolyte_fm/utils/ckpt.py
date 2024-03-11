from pathlib import Path

from pytorch_lightning.cli import LightningArgumentParser, SaveConfigCallback
from pytorch_lightning import Trainer, LightningModule


class SaveConfigWithCkpts(SaveConfigCallback):
    """Save Configuration with the model's checkpoints"""

    def __init__(
        self,
        parser: LightningArgumentParser,
        config,
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        super().__init__(
            parser,
            config,
            overwrite,
            multifile,
            save_to_log_dir=False,
        )

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        # Create the save log directory
        config_path = Path(trainer.log_dir or trainer.default_root_dir)
        config_filename = "config.yaml"
        if trainer.logger is not None:
            assert (
                trainer.logger.name is not None and trainer.logger.version is not None
            )
            config_path = config_path.joinpath(
                trainer.logger.name,
            )
            config_filename = str(trainer.logger.version) + ".yaml"

        config_path.mkdir(parents=True, exist_ok=True)

        # Save the config
        self.parser.save(
            self.config,
            config_path.joinpath(config_filename),
            skip_none=False,
            overwrite=self.overwrite,
            multifile=self.multifile,
        )
