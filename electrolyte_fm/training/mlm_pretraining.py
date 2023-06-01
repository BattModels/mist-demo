import os
from electrolyte_fm.models.roberta_base import RoBERTa
from pytorch_lightning import Trainer

here = os.path.abspath(os.path.dirname(__file__))

TOKENIZER_PATH = "../pretrained_tokenizers/ZINC_250k_BERT_loves_chemistry"
# TODO: Split test train validation
TRAIN_DATASET_PATH = os.path.join(here, "raw_data/250k_zinc.txt")
VAL_DATASET_PATH = os.path.join("raw_data/250k_zinc.txt")
TEST_DATASET_PATH = os.path.join("raw_data/250k_zinc.txt")
MAX_EPOCHS = 1
# GPUs per node
DEVICES = [0, 1, 2, 3, ]
NUM_NODES = 2

model = RoBERTa(
    pretrained_tokenizer_directory_path=TOKENIZER_PATH,
    train_dataset_path=TRAIN_DATASET_PATH,
    val_dataset_path=VAL_DATASET_PATH,
    test_dataset_path=TEST_DATASET_PATH,
)
trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu",
    devices=DEVICES,
    num_nodes=NUM_NODES)
trainer.fit(model)
