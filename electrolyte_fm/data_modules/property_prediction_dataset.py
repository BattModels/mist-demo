import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from datasets import load_dataset, Dataset
from datasets.distributed import split_dataset_by_node

import pytorch_lightning as pl
from statistics import mean
import torch
from torch.utils.data import DataLoader

from .data_utils import DataSetupMixin

TaskSpecs = List[Dict[str, Union[str, int]]]


class PropertyPredictionDataModule(pl.LightningDataModule, DataSetupMixin):
    def __init__(
            self, 
            path: str,
            tokenizer: str,
            dataset_name: str,
            task_specs: TaskSpecs,
            batch_size: int = 64,
            num_workers: int = 1,
            prefetch_factor: int = 4,
            val_batch_size: Optional[int] = None,
            train_dataset_length: Optional[int] = None,
            val_dataset_length: Optional[int] = None,
            test_dataset_length: Optional[int] = None,
    ):
        super().__init__()

        self.setup_tokenizer(tokenizer)
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

        ds = load_dataset(os.path.join(self.path, self.dataset_name))

        # Setup to partition datasets over ranks
        assert self.trainer is not None
        rank = self.trainer.global_rank or 0
        world_size = self.trainer.world_size or 1
        ds_train: Dataset = ds["train"].shuffle(seed=42)

        self.train_dataset: Dataset = split_dataset_by_node(
            ds_train,
            rank=rank,
            world_size=world_size,
        )
        self.val_dataset: Dataset = split_dataset_by_node(
            ds['validation'],
            rank=rank,
            world_size=world_size,
        )
        self.test_dataset: Dataset = split_dataset_by_node(
            ds['test'],
            rank=rank,
            world_size=world_size,
        )
        self.calculate_imputation_values()

    def calculate_imputation_values(self):
        for spec in self.task_specs:
            if spec.get("n_classes", 1) > 1:
                spec["fill_value"] = -1
            else:
                spec["fill_value"] = mean(
                    d for d in self.train_dataset[spec['measure_name']] if d is not None
                )

    def data_collator(self, batch):
        tokens = self.tokenizer._batch_encode_plus(
            [sample['smiles'] for sample in batch], 
            add_special_tokens=True,
            return_tensors="pt",
            padding_strategy = "longest"
            )
        for spec in self.task_specs:
                tokens[spec["measure_name"]] = torch.tensor([
                    sample[spec["measure_name"]] or spec["fill_value"] for sample in batch
                    ])
        
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
            shuffle=False,
        )

    def test_dataset(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )