import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# {"measure_name": {"task_type": "regression"}}
# {"measure_name": {"task_type": "classification", "n_classes": }}
TaskSpec = Dict[str, Union[str, int]]


class MultitaskDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            path: str,
            tokenizer: str,
            dataset_name: str,
            task_specs: List[TaskSpec],
            batch_size: int = 64,
            num_workers: int = 1,
            prefetch_factor: int = 4,
            val_batch_size: Optional[int] = None,
            train_dataset_length: Optional[int] = None,
            val_dataset_length: Optional[int] = None,
            test_dataset_length: Optional[int] = None,
    ):
        super().__init__()

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer,
            trust_remote_code=True,
            cache_dir=".cache",  # Cache Tokenizer in working directory
        )
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.path: Path = Path(path)
        assert self.path.is_dir() or self.path.is_file()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.dataset_name = dataset_name
        self.task_specs = task_specs
        self.train_dataset_length = train_dataset_length
        self.val_dataset_length = val_dataset_length
        self.test_dataset_length = test_dataset_length
        self.task_specs = task_specs
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        full_dataset = load_dataset(os.path.join(self.path, self.dataset_name))
        self.train_dataset = full_dataset['train']
        self.val_dataset = full_dataset['validation']
        self.test_dataset = full_dataset['validation']

    def data_collator(self, batch):
        tokens = self.tokenizer(
            [sample['smiles'] for sample in batch], 
            padding=True, 
            return_tensors="pt",
            add_special_tokens=True
            )
        tokens["targets"] = {
            measure_name: torch.tensor([
                sample[measure_name] for sample in batch
                ]) for measure_name in self.task_specs
            }
        return tokens
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False
        )

    def test_dataset(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )