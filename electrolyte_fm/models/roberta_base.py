from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
)


class RoBERTa(pl.LightningModule):
    """
    PyTorch Lightning module for RoBERTa model MLM pre-training.
    """

    def __init__(
        self,
        pretrained_tokenizer_directory_path: str,
        train_dataset_path: str,
        val_dataset_path: Optional[str],
        test_dataset_path: str,
        vocab_size: int = 52_000,
        max_position_embeddings: int = 514,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 6,
        batch_size: int = 64,
        num_workers: int = 64,
        learning_rate: float = 1e-4,
    ):
        super().__init__()

        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            pretrained_tokenizer_directory_path, max_len=512
        )

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        self.config = RobertaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=1,
        )
        self.model = RobertaForMaskedLM(config=self.config)

    def setup(self, stage):
        if not hasattr(self, "model"):
            self.model = RobertaForMaskedLM(config=self.config)

    def load_dataset(
        self, raw_dataset_path, block_size: int = 128
    ) -> LineByLineTextDataset:
        dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path=raw_dataset_path,
            block_size=block_size,
        )
        return dataset

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.load_dataset(self.train_dataset_path)
        dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.load_dataset(self.val_dataset_path)
        dataloader = DataLoader(
            val_dataset,
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return dataloader

    def forward(self, batch, **kwargs):
        out = self.model(
            batch["input_ids"],
            labels=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs,
        )
        return out

    def training_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for n, p in self.model.named_parameters()], lr=self.learning_rate
        )
        return optimizer
