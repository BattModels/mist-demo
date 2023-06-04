import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split
from transformers import LineByLineTextDataset, RobertaTokenizerFast

from electrolyte_fm.models.roberta_base import RoBERTa
from electrolyte_fm.utils.callbacks import ThroughputMonitor

here = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

TOKENIZER_PATH = os.path.join(here, "pretrained_tokenizers/ZINC_250k/")
DATASET_PATH = os.path.join(here, "raw_data/250k_zinc.txt")
MAX_EPOCHS = 4
NUM_NODES = int(os.environ["SLURM_NNODES"])
try:
    GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])
except KeyError:
    GPUS_PER_NODE = int(len(os.environ["SLURM_JOB_GPUS"].split(',')))

tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=512)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=DATASET_PATH,
    block_size=128,
)

train_dataset, test_dataset, val_dataset = random_split(
    dataset, lengths=[0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
)

model = RoBERTa(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    val_dataset=val_dataset,
    tokenizer_path=TOKENIZER_PATH,
)


wandb_logger = WandbLogger(
    name=f"{torch.cuda.get_device_name()}_NN_{NUM_NODES}_GPN_{GPUS_PER_NODE}",
    project="electrolyte-fm",
)
callbacks = [ThroughputMonitor(batch_size=64, num_nodes=NUM_NODES, wandb_active=True)]
trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu",
    devices=list(range(GPUS_PER_NODE)),
    num_nodes=NUM_NODES,
    logger=wandb_logger,
    callbacks=callbacks,
    # limit_train_batches=10,
    # limit_val_batches=5,
)
trainer.fit(model)
