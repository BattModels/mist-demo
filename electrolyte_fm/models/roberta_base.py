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
        tokenizer_path: str,
        train_dataset: LineByLineTextDataset,
        val_dataset: LineByLineTextDataset,
        test_dataset: LineByLineTextDataset,
        vocab_size: int = 52_000,
        max_position_embeddings: int = 512,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 6,
    ) -> None:
        super().__init__()

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            tokenizer_path, max_len=512
        )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=64,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, collate_fn=self.data_collator, batch_size=64
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataloader,
            collate_fn=self.data_collator,
            batch_size=64,
        )

    def forward(self, batch, **kwargs):  # type: ignore[override]
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
            [p for n, p in self.model.named_parameters()], lr=1e-4
        )
        return optimizer
