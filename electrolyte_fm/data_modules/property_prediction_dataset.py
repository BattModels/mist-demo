import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from datasets import load_dataset
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
        full_dataset = load_dataset(os.path.join(self.path, self.dataset_name))
        self.train_dataset = full_dataset['train']
        self.val_dataset = full_dataset['validation']
        self.test_dataset = full_dataset['test']
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
        tokens = self.tokenizer(
            [sample['smiles'] for sample in batch], 
            padding=True, 
            return_tensors="pt",
            add_special_tokens=True
            )
        
        for spec in self.task_specs:
                tokens[spec["measure_name"]] = torch.tensor([
                    sample[spec["measure_name"]] or spec["fill_value"] for sample in batch
                    ])
        return tokens
    
    def train_dataloader(self):
        sampler = torch.utils.data.DistributedSampler(
            self.train_dataset, shuffle=True)
        return DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        sampler = torch.utils.data.DistributedSampler(
            self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            sampler=sampler
        )

    def test_dataset(self):
        sampler = torch.utils.data.DistributedSampler(
            self.val_dataset, shuffle=False)
        return DataLoader(
            self.test_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            sampler=sampler
        )