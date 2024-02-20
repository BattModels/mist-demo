from os import cpu_count
from pathlib import Path

import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, random_split
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaTokenizerFast,
)


class RobertaDataSet(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_path: str,
        path: str,
        max_length: int = 512,
        mlm_probability=0.15,
        block_size: int = 128,
        batch_size: int = 64,
        val_batch_size=None,
    ):
        super().__init__()

        # Locate Tokeniser and dataset
        self.tokenizer_path: Path = Path(tokenizer_path)
        self.path: Path = Path(path)
        assert self.tokenizer_path.is_dir()
        assert self.path.is_dir() or self.path.is_file()

        self.max_length = max_length
        self.block_size = block_size
        self.mlm_probability = mlm_probability
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.save_hyperparameters()

    def prepare_data(self):
        self.__load_dataset()

    def __load_dataset(self, num_proc=None):
        if not hasattr(self, "dataset"):
            if num_proc is None:
                num_proc = cpu_count()

            dataset = load_dataset(
                str(self.path),
                cache_dir=".cache",
                num_proc=num_proc,
                keep_in_memory=False,
                split="train",
            )
            assert isinstance(dataset, Dataset)
            self.dataset = dataset

        return self.dataset

    def setup(self, stage: str) -> None:
        tokenizer = RobertaTokenizerFast.from_pretrained(
            self.tokenizer_path, max_len=self.max_length
        )
        dataset = self.__load_dataset(num_proc=16)
        dataset = dataset.with_transform(
            lambda batch: tokenizer(
                batch["text"], truncation=True, padding=True, return_tensors="pt"
            )
        )
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            lengths=[0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(42),
        )

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=self.mlm_probability,
            mlm=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataset(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
        )
