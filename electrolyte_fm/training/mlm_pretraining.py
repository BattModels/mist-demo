import os
from datetime import datetime
from electrolyte_fm.models.roberta_base import RoBERTa
from pytorch_lightning import Trainer
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger

here = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

TOKENIZER_PATH = os.path.join(here,"pretrained_tokenizers/ZINC_250k_BERT_loves_chemistry/")
# TODO: Split test train validation
TRAIN_DATASET_PATH = os.path.join(here, "raw_data/250k_zinc.txt")
VAL_DATASET_PATH = os.path.join(here, "raw_data/250k_zinc.txt")
TEST_DATASET_PATH = os.path.join(here, "raw_data/250k_zinc.txt")
MAX_EPOCHS = 1
# GPUs per node
DEVICES =  [0, 1, 2, 3, ]
NUM_NODES = 2

model = RoBERTa(
    pretrained_tokenizer_directory_path=TOKENIZER_PATH,
    train_dataset_path=TRAIN_DATASET_PATH,
    val_dataset_path=VAL_DATASET_PATH,
    test_dataset_path=TEST_DATASET_PATH,
)

if __name__=="__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"
    time_stamp = datetime.utcnow().strftime(format="%d%m%y_%H%M")
    profiler = AdvancedProfiler(dirpath=here, filename=f"perf_logs_{time_stamp}")
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=DEVICES,
        num_nodes=NUM_NODES,
        profiler=profiler,
        accumulate_grad_batches=8,
        callbacks=[DeviceStatsMonitor()],
        logger=WandbLogger(project="fm_electrolyte", log_model="all")
    )
    trainer.fit(model)
