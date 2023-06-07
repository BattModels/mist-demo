from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.cli import LightningCLI
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
    """ Ensure that Wandb only gets launcher on Rank-0 """:w
    return WandbLogger(project="electrolyte-fm")


def cli_main():
    callbacks = [ThroughputMonitor()]
    MyLightningCLI(
        RoBERTa,
        RobertaDataSet,
        trainer_defaults={
            "callbacks": callbacks,
            "logger": logger(),
        },
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli_main()
