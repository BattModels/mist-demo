from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from transformers import (DataCollatorForLanguageModeling,
                          LineByLineTextDataset, RobertaTokenizerFast)


class RobertaDataSet(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_path=None,
        dataset_path=None,
        structure_data: bool = False,
        max_length: int = 512,
        mlm_probability=0.15,
        block_size: int = 128,
        batch_size: int = 64,
        val_batch_size=None,
    ):
        super().__init__()
        self.tokenizer_dir: str = "ZINC_250k_XYZ" if structure_data else "ZINC_250k"

        self.tokenizer_path: Path = (
            Path(tokenizer_path)
            if tokenizer_path
            else Path(__file__).parent.parent.joinpath(
                "pretrained_tokenizers", self.tokenizer_dir
            )
        )

        self.data_file: str = "250k_zinc_xyz.txt" if structure_data else "250k_zinc.txt"
        self.dataset_path: Path = (
            Path(dataset_path)
            if dataset_path
            else Path(__file__).parent.parent.joinpath("raw_data", self.data_file)
        )
        self.max_length = max_length
        self.block_size = block_size
        self.mlm_probability = mlm_probability
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.save_hyperparameters()

    def setup(self, stage):
        tokenizer = RobertaTokenizerFast.from_pretrained(
            str(self.tokenizer_path), max_len=self.max_length
        )
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=str(self.dataset_path),
            block_size=self.block_size,
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
            num_workers=8,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=8,
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
