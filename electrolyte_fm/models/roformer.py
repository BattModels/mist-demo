import torch
import pytorch_lightning as pl
from transformers import (
    RoFormerConfig,
    RoFormerForMaskedLM,
)


class RoFormer(pl.LightningModule):
    """
    PyTorch Lightning module for RoFormer model MLM pre-training.
    """

    def __init__(
            self,
            vocab_size: int = 52_000,
            max_position_embeddings: int = 512,
            num_attention_heads: int = 12,
            num_hidden_layers: int = 6,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = RoFormerConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=1,
        )
        self.model = RoFormerForMaskedLM(config=self.config)

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

        self.log(
            "train/perplexity",
            torch.exp(loss.cpu().long()).item(),
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "val/perplexity",
            torch.exp(loss.cpu().long()).item(),
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
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

        self.log(
            "test/perplexity",
            torch.exp(loss.cpu().long()).item(),
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for n, p in self.model.named_parameters()], lr=1e-4
        )
        return optimizer