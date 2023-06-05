""" Custom callbacks for benchmarking, adapted from GenSLM
    https://github.com/ramanathanlab/genslm/blob/71beb030df72010f5a4883a1f1a0b25bbafbe4a8/genslm/utils.py
"""
import time
from statistics import mean
from typing import Any, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class ThroughputMonitor(Callback):
    """Custom callback in order to monitor the throughput and log to weights and biases."""

    def __init__(self) -> None:
        """Logs throughput statistics starting at the 2nd epoch."""
        super().__init__()
        self.start_time = 0.0
        self.batch_times: List[float] = []
        self.epoch_throughputs: List[float] = []
        self.epoch_sample_times: List[float] = []

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        if pl_module.current_epoch > 0:
            self.start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if pl_module.current_epoch > 0:
            batch_time = time.perf_counter() - self.start_time
            self.batch_times.append(batch_time)
            pl_module.logger.log_metrics({"stats/batch_time": batch_time})

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if pl_module.current_epoch > 0:
            # compute average epoch throughput
            avg_batch_time = mean(self.batch_times)
            macro_batch_size = trainer.world_size * trainer.datamodule.batch_size
            avg_epoch_throughput = macro_batch_size / avg_batch_time
            avg_secs_per_sample = avg_batch_time / macro_batch_size

            self.epoch_throughputs.append(avg_epoch_throughput)
            self.epoch_sample_times.append(avg_secs_per_sample)
            self.batch_times = []  # Reset for next epoch
            pl_module.logger.log_metrics(
                {
                    "stats/avg_epoch_throughput": avg_epoch_throughput,
                    "stats/avg_secs_per_sample": avg_secs_per_sample,
                }
            )

    @property
    def average_throughput(self):
        if len(self.epoch_throughputs) > 1:
            return mean(self.epoch_throughputs)
        elif self.epoch_throughputs:
            return self.epoch_throughputs[0]
        else:
            return float("nan")

    @property
    def average_sample_time(self):
        if len(self.epoch_sample_times) > 1:
            return mean(self.epoch_sample_times)
        elif self.epoch_sample_times:
            return self.epoch_sample_times[0]
        return float("nan")

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # Collect metrics on each rank and compute overall statistics on rank 0
        metrics = self.average_throughput, self.average_sample_time
        trainer._accelerator_connector.strategy.barrier()
        metrics = pl_module.all_gather(metrics)
        throughputs, sample_times = metrics[0], metrics[1]
        if trainer.is_global_zero:
            trainer.logger.log_metrics(
                {
                    "stats/throughput_avg": throughputs.mean().item(),
                    "stats/throughput_stdev": throughputs.std().item(),
                    "stats/sample_time_avg": sample_times.mean().item(),
                    "stats/sample_time_stdev": sample_times.std().item(),
                    "stats/macro_batch_size": trainer.datamodule.batch_size
                    * trainer.world_size,
                    "stats/ranks": trainer.world_size,
                }
            )
