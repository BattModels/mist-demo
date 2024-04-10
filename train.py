import os
from datetime import timedelta

import torch
from jsonargparse import lazy_instance
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from electrolyte_fm.utils.callbacks import ThroughputMonitor

# classes passed via cli
from electrolyte_fm.models.roberta_base import RoBERTa
from electrolyte_fm.models.roberta_dataset import RobertaDataSet
from electrolyte_fm.models.lm_classification import LMClassification
from electrolyte_fm.models.lm_regression import LMRegression
from electrolyte_fm.models.property_prediction_dataset import \
    PropertyPredictionDataModule
from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts

class MyLightningCLI(LightningCLI):
    def before_fit(self):
        if logger := self.trainer.logger:
            logger.log_hyperparams(
                {
                    "n_gpus_per_node": self.trainer.num_devices,
                    "n_nodes": self.trainer.num_nodes,
                    "world_size": self.trainer.world_size,
                }
            )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # Set model vocab_size from the dataset's vocab size
        parser.link_arguments(
            "data.vocab_size", "model.init_args.vocab_size", apply_on="instantiate"
        )


def cli_main(args=None):
    monitor = "val/loss_epoch"
    callbacks = [
        ThroughputMonitor(),
        ModelCheckpoint(
            save_last="link",
            filename="epoch={epoch}-step={step}-val_loss={" + monitor + ":.2f}",
            monitor=monitor,
            save_top_k=5,
            verbose=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}",
            save_top_k=2,
            monitor="step",
            verbose=True,
            mode="max",
            train_time_interval=timedelta(minutes=30),
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor("step"),
    ]

    num_nodes = int(os.environ.get("NRANKS", 1))
    rank = int(os.environ.get(
        "PMI_RANK", os.environ.get("GLOBAL_RANK", 0)
        ))
    os.environ["NODE_RANK"] = str(rank % num_nodes)
    os.environ["GLOBAL_RANK"] = str(rank % num_nodes)

    # num_nodes = int(os.environ.get("NRANKS", 1))
    # rank = int(os.environ.get("PMI_RANK", 1))
    # os.environ["NODE_RANK"] = str(rank % num_nodes)
    # os.environ["GLOBAL_RANK"] = str(rank % num_nodes)
    print(f"PY: NUM_NODES: {num_nodes} PMI_RANK: {rank} PID {os.getpid()}")
    if rank is not None and int(rank) != 0:
        logger = None
    else:
        logger = lazy_instance(
            WandbLogger, project="mist", save_code=True, tags=["pretraining"]
        )

    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    return MyLightningCLI(
        trainer_defaults={
            "callbacks": callbacks,
            "logger": logger,
            "precision": "16-mixed",
            "devices": -1,
            "num_nodes": num_nodes or 1,
            "strategy": "deepspeed",
            "use_distributed_sampler": False,  # Handled by DataModule (Needed as Iterable)
        },
        save_config_callback=SaveConfigWithCkpts,
        save_config_kwargs={"overwrite": True},
        args=args,
        run=args is None,  # support unit testing
    )


if __name__ == "__main__":
    seed_everything(42, workers=True)
    cli_main()
