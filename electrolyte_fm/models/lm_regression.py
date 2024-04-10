import pytorch_lightning as pl
from pytorch_lightning.cli import OptimizerCallable, LRSchedulerCallable
import torch

from .model_utils import DeepSpeedMixin
from .prediction_task_head import PredictionTaskHead
from .roberta_base import RoBERTa


class LMRegression(pl.LightningModule, DeepSpeedMixin):
    """
    PyTorch Lightning module for finetuning LM encoder model on classification 
    tasks.
    """

    def __init__(
        self,
        pretrained_checkpoint: str,
        encoder_class: str = "roberta",
        freeze_encoder: bool = False,
        learning_rate: float = 1.6e-4,
        num_targets: int = 1, 
        dropout: float = 0.2,
        optimizer: OptimizerCallable = torch.optim.AdamW, 
        lr_schedule: LRSchedulerCallable | None = None,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.encoder_class = encoder_class
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.save_hyperparameters(ignore=["optimizer", "lr_schedule"])

        # Expose encoder
        self.encoder = RoBERTa.load(checkpoint_dir= pretrained_checkpoint).model.roberta
        self.task_network = PredictionTaskHead(
            embed_dim=self.encoder.config.hidden_size, 
            output_size=num_targets,
            dropout=self.dropout
        )
        self.loss = torch.nn.MSELoss()
        self.freeze_encoder = freeze_encoder
    
    def forward(self, batch, **kwargs):  # type: ignore[override]
        embedding = self.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs,
            return_dict=False
        )
        
        sequence_output = embedding[0]
        out = self.task_network(sequence_output)
        return out

    def training_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = self.loss(outputs, batch['targets'])
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
        loss = self.loss(outputs, batch['targets'])
        self.log(
            "val/loss", 
            loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = self.loss(outputs, batch['targets'])
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
        learnable_params = [p for _, p in self.task_network.named_parameters()]
        if not self.freeze_encoder:
            learnable_params.extend([p for _, p in self.encoder.named_parameters()])

        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer