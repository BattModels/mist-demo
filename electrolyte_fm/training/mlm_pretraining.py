import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from transformers import LineByLineTextDataset, RobertaTokenizerFast

from electrolyte_fm.models.roberta_base import RoBERTa
from electrolyte_fm.utils.callbacks import ThroughputMonitor

here = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

TOKENIZER_PATH = os.path.join(
    here, "pretrained_tokenizers/ZINC_250k/"
)
# TODO: Split test train validation
TRAIN_DATASET_PATH = os.path.join(here, "raw_data/250k_zinc.txt")
VAL_DATASET_PATH = os.path.join(here, "raw_data/250k_zinc.txt")
TEST_DATASET_PATH = os.path.join(here, "raw_data/250k_zinc.txt")
MAX_EPOCHS = 1
# GPUs per node
NUM_NODES = int(os.environ["SLURM_NNODES"])
GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])

tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=512)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=TRAIN_DATASET_PATH,
    block_size=128,
)

model = RoBERTa(
    train_dataset=dataset,
    test_dataset=dataset,
    val_dataset=dataset,
    tokenizer_path=TOKENIZER_PATH,
)


wandb_logger = WandbLogger(
    name=f"{torch.cuda.get_device_name()}_NN_{NUM_NODES}_GPN_{GPUS_PER_NODE}",
    project="electrolyte-fm",
)
callbacks = [ThroughputMonitor(batch_size=64, num_nodes=NUM_NODES, wandb_active=True)]
trainer = Trainer(
    max_epochs=4,
    accelerator="gpu",
    devices=list(range(GPUS_PER_NODE)),
    num_nodes=NUM_NODES,
    logger=wandb_logger,
    callbacks=callbacks,
)
trainer.fit(model)
