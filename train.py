from os import walk

from lightning_fabric import plugins
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import MPIEnvironment
from electrolyte_fm.models.roberta_base import RoBERTa
from electrolyte_fm.models.dataset import RobertaDataSet
from electrolyte_fm.utils.callbacks import ThroughputMonitor


class MyLightningCLI(LightningCLI):
    def before_fit(self):
        self.trainer.logger.log_hyperparams(
            {
                "n_gpus_per_node": self.trainer.num_devices,
                "n_nodes": self.trainer.num_nodes,
            }
        )


def detect_mpi(trainer_defaults: dict):
    mpienv = MPIEnvironment()
    if not mpienv.detect():
        return trainer_defaults
    print(
        {
            "WORLD_SIZE": mpienv.world_size(),
            "GLOBAL_RANK": mpienv.global_rank(),
            "LOCAL_RANK": mpienv.local_rank(),
            "MAIN_ADDRESS": mpienv.main_address,
            "MAIN_PORT": mpienv.main_port,
        }
    )
    num_gpus_per_node = 4
    num_nodes = int(mpienv.world_size() / num_gpus_per_node)
    trainer_defaults["num_nodes"] = num_nodes
    trainer_defaults["devices"] = num_gpus_per_node
    trainer_defaults["plugins"] = mpienv
    return trainer_defaults


def logger():
    return WandbLogger(project="electrolyte-fm")


def cli_main():
    callbacks = [
        ThroughputMonitor(),
        EarlyStopping(monitor="val/perplexity"),
    ]
    trainer_defaults = {
        "callbacks": callbacks,
        "precision": "16-mixed",
        "strategy": "ddp",
    }
    trainer_defaults = detect_mpi(trainer_defaults)

    # Decide if we should load wandb
    if "plugins" in trainer_defaults:
        mpienv = trainer_defaults["plugins"]
        if mpienv.global_rank() == 0:
            trainer_defaults["logger"] = logger()
    else:
        trainer_defaults["logger"] = logger()

    MyLightningCLI(
        RoBERTa,
        RobertaDataSet,
        trainer_defaults=trainer_defaults,
        save_config_callback=None,
    )


if __name__ == "__main__":
    seed_everything(42, workers=True)
    cli_main()
