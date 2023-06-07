from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from transformers import (
    LineByLineTextDataset,
    RoFormerTokenizer,
    DataCollatorForLanguageModeling,
)


class RoFormerDataSet(pl.LightningDataModule):
    def __init__(
            self,
            dataset_path=None,
            vocab_filepath=None,
            structure_data: bool = False,
            max_length: int = 512,
            mlm_probability=0.15,
            block_size: int = 128,
            batch_size: int = 64,
            val_batch_size=None,
    ):
        super().__init__()

        self.datafile: str = "250k_zinc_xyz.txt" if structure_data else "250k_zinc.txt"
        self.dataset_path: Path = (
            Path(dataset_path)
            if dataset_path
            else Path(__file__).parent.parent.joinpath("raw_data", self.datafile)
        )

        self.vocab_file: str = "zinc250k_xyz_vocab.txt" if structure_data else "zinc250k_vocab.txt"
        self.vocab_filepath: Path = (
            Path(vocab_filepath)
            if vocab_filepath
            else Path(__file__).parent.parent.joinpath("raw_data", self.vocab_file)

        )

        self.max_length = max_length
        self.block_size = block_size
        self.mlm_probability = mlm_probability
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.save_hyperparameters()

    def setup(self, stage):
        tokenizer = RoFormerTokenizer(vocab_file=self.vocab_file, do_basic_tokenize=False)
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
        )

    def train_dataset(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
        )