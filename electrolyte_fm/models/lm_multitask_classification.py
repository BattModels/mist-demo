from typing import Dict, List, Union

import pytorch_lightning as pl
from pytorch_lightning.cli import OptimizerCallable, LRSchedulerCallable
import torch

from .prediction_task_head import PredictionTaskHead
from .model_utils import DeepSpeedMixin
from .roberta_base import RoBERTa

TaskSpecs = List[Dict[str, Union[str, int]]]

class LMMultiTaskClassification(pl.LightningModule, DeepSpeedMixin):
    """
    PyTorch Lightning module for finetuning LM encoder model on multiple tasks.
    """

    def __init__(
        self,
        pretrained_checkpoint: str,
        task_specs: TaskSpecs, 
        encoder_class: pl.LightningModule = RoBERTa,
        freeze_encoder: bool = False,
        learning_rate: float = 1.6e-4,
        dropout: float = 0.2,
        optimizer: OptimizerCallable = torch.optim.AdamW, 
        lr_schedule: LRSchedulerCallable | None = None
    ) -> None:
        
        super().__init__()

        self.learning_rate = learning_rate
        self.dropout = dropout
        self.encoder_class = encoder_class
        self.task_specs = task_specs
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule

        # Expose encoder
        self.encoder = RoBERTa.load(checkpoint_dir= pretrained_checkpoint).model.roberta

        self.save_hyperparameters()

        head_hyperparams = {
            "embed_dim": self.encoder.config.hidden_size,
            "dropout": self.dropout
        }
        
        self.task_networks = torch.nn.ModuleDict(
            { 
                spec["measure_name"]: PredictionTaskHead(
                **head_hyperparams,
                output_size=spec.get("n_classes", 2)) for spec in self.task_specs
            }
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.freeze_encoder = freeze_encoder
    
    def forward(self, batch, **kwargs):  # type: ignore[override]
        embedding = self.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs,
            return_dict=False
        )
        
        sequence_output = embedding[0]
        out = {
            spec["measure_name"]: self.task_networks[
                spec["measure_name"]
                ](sequence_output)
            for spec in self.task_specs
        }
        return out

    def training_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)

        losses = 0
        for spec in self.task_specs:
            target = spec["measure_name"]
            losses += self.loss(outputs[target], batch[target])
        mean_loss = losses/len(self.task_specs)

        self.log(
            "train/loss",
            mean_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return mean_loss

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        
        losses = 0
        for spec in self.task_specs:
            target = spec["measure_name"]
            losses += self.loss(outputs[target], batch[target])
        mean_loss = losses/len(self.task_specs)

        self.log(
            "val/loss", 
            mean_loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            sync_dist=True
        )
        return mean_loss

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)

        losses = 0
        for spec in self.task_specs:
            target = spec["measure_name"]
            losses += self.loss(outputs[target], batch[target])
        mean_loss = losses/len(self.task_specs)

        self.log(
            "test/loss",
            mean_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return mean_loss
    
    def configure_optimizers(self):
        learnable_params = [
            p for _, task_network in self.task_networks.items() for _, p in task_network.named_parameters()
            ]
        if not self.freeze_encoder:
            learnable_params.extend([p for _, p in self.encoder.named_parameters()])

        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
