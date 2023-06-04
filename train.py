import os
import electrolyte_fm

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split
from transformers import LineByLineTextDataset, RobertaTokenizerFast

from electrolyte_fm.models.roberta_base import RoBERTa
from electrolyte_fm.utils.callbacks import ThroughputMonitor

here = os.path.abspath(os.path.dirname(electrolyte_fm.__file__))

TOKENIZER_PATH = os.path.join(here, "pretrained_tokenizers/ZINC_250k/")
DATASET_PATH = os.path.join(here, "raw_data/250k_zinc.txt")
MAX_EPOCHS = 4
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


wandb_logger = WandbLogger(project="electrolyte-fm")
callbacks = [ThroughputMonitor(batch_size=64)]
trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu",
    logger=wandb_logger,
    callbacks=callbacks,
    limit_train_batches=10,
    limit_val_batches=10,
)

trainer.logger.log_hyperparams(
    {"n_gpus_per_node": trainer.num_devices, "n_nodes": trainer.num_nodes}
)
trainer.fit(model)
