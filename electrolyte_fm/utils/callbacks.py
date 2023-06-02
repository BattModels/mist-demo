import time
from statistics import mean
from typing import Any, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class ThroughputMonitor(Callback):
    """
    Custom callback in order to monitor the throughput and log to weights and biases.
    """

    def __init__(
        self, batch_size: int, num_nodes: int = 1, wandb_active: bool = True
    ) -> None:
        """Logs throughput statistics starting at the 2nd epoch."""
        super().__init__()
        self.wandb_active = wandb_active
        self.start_time = 0.0
        self.average_throughput = 0.0
        self.average_sample_time = 0.0
        self.batch_times: List[float] = []
        self.epoch_throughputs: List[float] = []
        self.epoch_sample_times: List[float] = []
        self.num_ranks = num_nodes * torch.cuda.device_count()
        self.macro_batch_size = batch_size * self.num_ranks

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        if pl_module.current_epoch > 0:
            self.start_time = time.time()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if pl_module.current_epoch > 0:
            batch_time = time.time() - self.start_time
            self.batch_times.append(batch_time)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if pl_module.current_epoch > 0:
            # compute average epoch throughput
            avg_batch_time = mean(self.batch_times)
            avg_epoch_throughput = self.macro_batch_size / avg_batch_time
            avg_secs_per_sample = avg_batch_time / self.macro_batch_size
            # pl_module.log("stats/average_epoch_throughput", avg_epoch_throughput)
            # pl_module.log("stats/average_secs_per_sample", avg_secs_per_sample)
            self.epoch_throughputs.append(avg_epoch_throughput)
            self.epoch_sample_times.append(avg_secs_per_sample)
            self.batch_times = []  # Reset for next epoch

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.average_throughput = mean(self.epoch_throughputs)
        self.average_sample_time = mean(self.epoch_sample_times)

        # Collect metrics on each rank and compute overall statistics on rank 0
        metrics = self.average_throughput, self.average_sample_time
        trainer._accelerator_connector.strategy.barrier()
        metrics = pl_module.all_gather(metrics)
        throughputs, sample_times = metrics[0], metrics[1]
        if trainer.is_global_zero:
            thru_avg, thru_stdev = throughputs.mean().item(), throughputs.std().item()
            print(
                f"\nAVERAGE THROUGHPUT: {thru_avg} +- {thru_stdev} "
                f" samples/second over {self.num_ranks} ranks"
            )

            sample_time_avg = sample_times.mean().item()
            sample_time_stdev = sample_times.std().item()

            print(
                f"AVERAGE SECONDS PER SAMPLE: {sample_time_avg} +- {sample_time_stdev} "
                f"seconds/sample over {self.num_ranks} ranks"
            )

            if self.wandb_active:
                pl_module.logger.log_text(
                    key="stats/performance",
                    columns=[
                        "throughput_avg",
                        "throughput_stdev",
                        "sample_time_avg",
                        "sample_time_stdev",
                        "macro_batch_size",
                        "ranks",
                    ],
                    data=[
                        [
                            thru_avg,
                            thru_stdev,
                            sample_time_avg,
                            sample_time_stdev,
                            self.macro_batch_size,
                            self.num_ranks,
                        ]
                    ],
                )
