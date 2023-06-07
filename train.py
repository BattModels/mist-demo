import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import MPIEnvironment
from electrolyte_fm.models.roberta_base import RoBERTa
from electrolyte_fm.models.dataset import RobertaDataSet
from electrolyte_fm.utils.callbacks import ThroughputMonitor
from electrolyte_fm.utils.decorator import leader_only


class MyLightningCLI(LightningCLI):
    def before_fit(self):
        self.trainer.logger.log_hyperparams(
            {
                "n_gpus_per_node": self.trainer.num_devices,
                "n_nodes": self.trainer.num_nodes,
            }
        )

@leader_only
def logger():
    """ Ensure that Wandb only gets launcher on Rank-0 """
    return WandbLogger(project="electrolyte-fm")


def cli_main():
    callbacks = [ThroughputMonitor()]
    mpienv = MPIEnvironment()
    print({
        "WORLD_SIZE": mpienv.world_size(),
        "GLOBAL_RANK": mpienv.global_rank(),
        "LOCAL_RANK": mpienv.local_rank(),
        "MAIN_ADDRESS": mpienv.main_address,
        "MAIN_PORT": mpienv.main_port,
    })
    num_gpus_per_node = 4
    num_nodes = int(mpienv.world_size() / num_gpus_per_node)
    if mpienv.global_rank() == 0:
        logger = WandbLogger(project="electrolyte-fm")
    else:
        logger = None

    MyLightningCLI(
        RoBERTa,
        RobertaDataSet,
        trainer_defaults={
            "callbacks": callbacks,
            "logger": logger,
            "plugins": mpienv,
            "precision": 16,
            "devices": num_gpus_per_node,
            "num_nodes": num_nodes,
            "strategy": "ddp",
        },
        save_config_callback=None,
    )


if __name__ == "__main__":
    seed_everything(42, workers=True)
    cli_main()
