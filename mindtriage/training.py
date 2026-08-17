import atexit
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


def timestamped_run_dir(base_dir: Path, name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base_dir / "runs" / name / timestamp


class TrainingMonitor:
    """Terminal percentage/ETA reporting plus TensorBoard scalar logging."""

    def __init__(
        self,
        run_dir: Path,
        task_name: str,
        epochs: int,
        steps_per_epoch: int,
        log_every_steps: int = 1,
        tensorboard_enabled: bool = True,
    ):
        if epochs <= 0 or steps_per_epoch <= 0:
            raise ValueError("Training monitor requires positive epochs and steps per epoch")
        if log_every_steps <= 0:
            raise ValueError("log_every_steps must be greater than zero")
        self.run_dir = Path(run_dir)
        self.task_name = task_name
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.log_every_steps = log_every_steps
        self.global_step = 0
        self.started_at = time.monotonic()
        self.writer = None
        if tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "TensorBoard is not installed. Run `python -m pip install -r requirements.txt` "
                    "or use --no-tensorboard."
                ) from exc
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=str(self.run_dir), max_queue=10, flush_secs=1
            )
            self.writer.add_text("run/task", task_name, 0)
        atexit.register(self.close)
        print(
            json.dumps(
                {
                    "monitor": task_name,
                    "epochs": epochs,
                    "batches_per_epoch": steps_per_epoch,
                    "total_training_steps": self.total_steps,
                    "tensorboard": str(self.run_dir) if self.writer else "disabled",
                }
            ),
            flush=True,
        )

    def training_batches(self, loader, epoch: int):
        return tqdm(
            loader,
            total=len(loader),
            desc=f"{self.task_name} epoch {epoch}/{self.epochs}",
            unit="batch",
            mininterval=1.0,
            dynamic_ncols=True,
        )

    def evaluation_batches(self, loader, phase: str, epoch: int | None = None):
        suffix = f" epoch {epoch}/{self.epochs}" if epoch is not None else ""
        return tqdm(
            loader,
            total=len(loader),
            desc=f"{self.task_name} {phase}{suffix}",
            unit="batch",
            mininterval=1.0,
            dynamic_ncols=True,
            leave=False,
        )

    def log_batch(self, progress, epoch: int, batch_number: int, loss: float) -> float:
        self.global_step += 1
        completion = min(100.0, self.global_step * 100.0 / self.total_steps)
        elapsed = max(time.monotonic() - self.started_at, 0.001)
        seconds_per_step = elapsed / self.global_step
        remaining_seconds = max(
            0, round((self.total_steps - self.global_step) * seconds_per_step)
        )
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        overall_eta = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        progress.set_postfix(
            loss=f"{loss:.4f}",
            overall=f"{completion:.2f}%",
            overall_eta=overall_eta,
            refresh=False,
        )
        if self.writer and (
            self.global_step == 1 or self.global_step % self.log_every_steps == 0
        ):
            self.writer.add_scalar("train/batch_loss", loss, self.global_step)
            self.writer.add_scalar(
                "progress/completion_percent", completion, self.global_step
            )
            self.writer.add_scalar("progress/epoch", epoch, self.global_step)
            self.writer.add_scalar("progress/batch_in_epoch", batch_number, self.global_step)
        return completion

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        if not self.writer:
            return
        for name, value in metrics.items():
            self.writer.add_scalar(f"epoch/{name}", float(value), epoch)
        self.writer.flush()

    def log_test(self, metrics: dict[str, float]) -> None:
        if not self.writer:
            return
        for name, value in metrics.items():
            self.writer.add_scalar(f"test/{name}", float(value), self.global_step)
        self.writer.flush()

    def finish(self, epochs_completed: int) -> None:
        stopped_early = epochs_completed < self.epochs
        if self.writer:
            self.writer.add_scalar(
                "progress/completion_percent", 100.0, self.global_step + 1
            )
            self.writer.add_text(
                "run/completion_reason",
                "early_stopping" if stopped_early else "all_epochs_completed",
                self.global_step + 1,
            )
            self.writer.flush()
        print(
            json.dumps(
                {
                    "monitor": self.task_name,
                    "status": "complete",
                    "completion_percent": 100.0,
                    "epochs_completed": epochs_completed,
                    "epochs_planned": self.epochs,
                    "early_stopping": stopped_early,
                }
            ),
            flush=True,
        )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
