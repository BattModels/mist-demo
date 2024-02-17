import os
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from electrolyte_fm.utils.callbacks import ThroughputMonitor

# classes passed via cli
from electrolyte_fm.models.roberta_base import RoBERTa
from electrolyte_fm.models.roformer import RoFormer
from electrolyte_fm.models.roberta_dataset import RobertaDataSet
from electrolyte_fm.models.roformer_dataset import RoFormerDataSet


class MyLightningCLI(LightningCLI):
    def before_fit(self):
        self.trainer.logger.log_hyperparams(
            {
                "n_gpus_per_node": self.trainer.num_devices,
                "n_nodes": self.trainer.num_nodes,
                "world_size": self.trainer.world_size,
            }
        )


def cli_main():
    callbacks = [ThroughputMonitor(), EarlyStopping(monitor="val/perplexity")]

    num_nodes = os.environ.get("NRANKS")
    rank = os.environ.get("PMI_RANK")
    print(f"PY: NUM_NODES: {num_nodes} PMI_RANK: {rank} PID {os.getpid()}")
    if rank is not None and int(rank) == 0:
        logger = WandbLogger(project="mist")
    else:
        logger = None

    torch.set_num_threads(8)
    MyLightningCLI(
        trainer_defaults={
            "callbacks": callbacks,
            "logger": logger,
            "precision": "16-mixed",
            "devices": -1,
            "num_nodes": num_nodes,
            "strategy": "deepspeed",
        },
        save_config_callback=None,
    )


if __name__ == "__main__":
    seed_everything(42, workers=True)
    cli_main()
