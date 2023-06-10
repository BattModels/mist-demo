import os
import torch
import torch.multiprocessing as mp
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import  DeepSpeedStrategy
#from pytorch_lightning.plugins.environments import MPIEnvironment
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
    #mpienv = MPIEnvironment()
    #print({
    #    "WORLD_SIZE": mpienv.world_size(),
    #    "GLOBAL_RANK": mpienv.global_rank(),
    #    "LOCAL_RANK": mpienv.local_rank(),
    #    "MAIN_ADDRESS": mpienv.main_address,
    #    "MAIN_PORT": mpienv.main_port,
    #})
    with open("hellow", "w+") as fid:
        fid.write("hellow world")

    num_gpus_per_node = 4

    # Not the normal "World Size", Lightning's notion of world size
    # num_nodes = mpienv.world_size()
    num_nodes = 2
    print(f"Number of Nodes: {num_nodes}")
    rank = os.environ.get("PMI_RANK")
    print(f"ENV RANK GREP ME: {rank}, PID: {os.getpid()}")
    if rank is not None and int(rank) == 0:
        logger = WandbLogger(project="electrolyte-fm")
    else:
        logger = None

    torch.set_num_threads(8)
    MyLightningCLI(
        RoBERTa,
        RobertaDataSet,
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
    #mp.set_start_method("spawn")
    cli_main()
