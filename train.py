from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.cli import LightningCLI
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


def cli_main():
    callbacks = [ThroughputMonitor(), EarlyStopping(monitor='val/perplexity')]
    MyLightningCLI(
        RoBERTa,
        RobertaDataSet,
        trainer_defaults={
            "callbacks": callbacks,
            "logger": WandbLogger(project="electrolyte-fm"),
        },
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli_main()
