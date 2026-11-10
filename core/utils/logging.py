# Structured Logging Utilities
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, TextIO
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
import threading

import torch


@dataclass
class LogRecord:
    """Structured log record."""
    timestamp: str
    level: str
    logger: str
    message: str
    # Additional fields
    step: Optional[int] = None
    episode: Optional[int] = None
    agent: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(',', ':'), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'LogRecord':
        return cls(**json.loads(json_str))


class JSONLHandler(logging.Handler):
    """Logging handler that writes JSONL format."""

    def __init__(self, file_path: Union[str, Path], mode: str = 'a', buffering: int = 1):
        super().__init__()
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file: TextIO = open(self.file_path, mode, buffering=buffering, encoding='utf-8')
        self.lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_record = LogRecord(
                timestamp=datetime.fromtimestamp(record.created).isoformat(),
                level=record.levelname,
                logger=record.name,
                message=record.getMessage(),
                step=getattr(record, 'step', None),
                episode=getattr(record, 'episode', None),
                agent=getattr(record, 'agent', None),
                metrics=getattr(record, 'metrics', {}),
                extra={k: v for k, v in record.__dict__.items()
                       if k not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                                  'levelname', 'levelno', 'lineno', 'module', 'msecs',
                                  'message', 'msg', 'name', 'pathname', 'process',
                                  'processName', 'relativeCreated', 'thread', 'threadName',
                                  'exc_info', 'exc_text', 'stack_info', 'step', 'episode',
                                  'agent', 'metrics']}
            )

            with self.lock:
                self.file.write(log_record.to_json() + '\n')
                self.file.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self.lock:
            self.file.close()
        super().close()


class TensorBoardLogger:
    """TensorBoard logging wrapper."""

    def __init__(self, log_dir: Union[str, Path], flush_secs: int = 30):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flush_secs = flush_secs
        self._writer = None
        self._step = 0

    @property
    def writer(self):
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(str(self.log_dir), flush_secs=self.flush_secs)
        return self._writer

    def set_step(self, step: int) -> None:
        self._step = step

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        step = step if step is not None else self._step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: Optional[int] = None) -> None:
        step = step if step is not None else self._step
        self.writer.add_scalars(tag, values, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None) -> None:
        step = step if step is not None else self._step
        self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple) -> None:
        dummy_input = torch.randn(*input_shape)
        self.writer.add_graph(model, dummy_input)

    def log_video(self, tag: str, video_tensor: torch.Tensor, step: Optional[int] = None, fps: int = 30) -> None:
        step = step if step is not None else self._step
        # video_tensor: [T, C, H, W] or [N, T, C, H, W]
        self.writer.add_video(tag, video_tensor, step, fps=fps)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None


class WandbLogger:
    """Weights & Biases logging wrapper."""

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        enabled: bool = True
    ):
        self.enabled = enabled
        self._run = None

        if enabled:
            try:
                import wandb
                self._run = wandb.init(
                    project=project,
                    entity=entity,
                    config=config,
                    tags=tags,
                    group=group,
                    name=name,
                    reinit=True
                )
            except ImportError:
                print("wandb not installed, disabling")
                self.enabled = False

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self._run:
            import wandb
            wandb.log(data, step=step)

    def log_model(self, path: str, name: str = "model") -> None:
        if self.enabled and self._run:
            import wandb
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(path)
            self._run.log_artifact(artifact)

    def finish(self) -> None:
        if self.enabled and self._run:
            import wandb
            wandb.finish()


def setup_logging(
    log_dir: Union[str, Path],
    level: str = "INFO",
    console: bool = True,
    jsonl: bool = True,
    tensorboard: bool = True,
    wandb: bool = False,
    wandb_config: Optional[Dict[str, Any]] = None,
    name: str = "ppo_training"
) -> Dict[str, Any]:
    """
    Setup comprehensive logging.

    Returns dict with logger instances for different backends.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Formatters
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(console_handler)

    # JSONL file handler
    jsonl_handler = None
    if jsonl:
        jsonl_path = log_dir / f"{name}_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        jsonl_handler = JSONLHandler(jsonl_path)
        jsonl_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(jsonl_handler)

    # TensorBoard
    tb_logger = None
    if tensorboard:
        tb_logger = TensorBoardLogger(log_dir / "tensorboard")

    # WandB
    wb_logger = None
    if wandb and wandb_config:
        wb_logger = WandbLogger(**wandb_config)

    return {
        "root": root_logger,
        "jsonl_handler": jsonl_handler,
        "tensorboard": tb_logger,
        "wandb": wb_logger,
        "log_dir": log_dir,
    }


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


@contextmanager
def log_context(logger: logging.Logger, **context):
    """Context manager to add context to all log records."""
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        for k, v in context.items():
            setattr(record, k, v)
        return record

    logging.setLogRecordFactory(record_factory)
    try:
        yield
    finally:
        logging.setLogRecordFactory(old_factory)


class MetricsLogger:
    """High-level metrics logging for training."""

    def __init__(
        self,
        loggers: Dict[str, Any],
        agent_name: str,
        log_interval: int = 1000
    ):
        self.loggers = loggers
        self.agent_name = agent_name
        self.log_interval = log_interval
        self.step = 0
        self.episode = 0
        self._buffer: Dict[str, list] = {}
        self._start_time = time.time()

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        step = step if step is not None else self.step

        # TensorBoard
        if self.loggers.get("tensorboard"):
            self.loggers["tensorboard"].log_scalar(f"{self.agent_name}/{name}", value, step)

        # WandB
        if self.loggers.get("wandb"):
            self.loggers["wandb"].log({f"{self.agent_name}/{name}": value}, step=step)

        # Buffer for periodic JSONL logging
        if name not in self._buffer:
            self._buffer[name] = []
        self._buffer[name].append(value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = "") -> None:
        for name, value in metrics.items():
            full_name = f"{prefix}/{name}" if prefix else name
            self.log_metric(full_name, value, step)

    def log_episode(self, episode_reward: float, episode_length: int, episode_info: Dict[str, Any]) -> None:
        self.episode += 1

        metrics = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            **episode_info
        }

        self.log_metrics(metrics, prefix="train")

        # JSONL logging via root logger
        root_logger = self.loggers.get("root")
        if root_logger:
            root_logger.info(
                f"Episode {self.episode} completed",
                extra={
                    "step": self.step,
                    "episode": self.episode,
                    "agent": self.agent_name,
                    "metrics": metrics
                }
            )

    def log_training_step(self, metrics: Dict[str, float]) -> None:
        self.step += 1
        self.log_metrics(metrics, prefix="train")

        # Periodic flush
        if self.step % self.log_interval == 0:
            self.flush()

    def log_eval(self, metrics: Dict[str, float]) -> None:
        self.log_metrics(metrics, prefix="eval")

        root_logger = self.loggers.get("root")
        if root_logger:
            root_logger.info(
                f"Evaluation completed",
                extra={
                    "step": self.step,
                    "agent": self.agent_name,
                    "metrics": metrics
                }
            )

    def flush(self) -> None:
        if self.loggers.get("tensorboard"):
            self.loggers["tensorboard"].flush()

    def close(self) -> None:
        self.flush()
        if self.loggers.get("tensorboard"):
            self.loggers["tensorboard"].close()
        if self.loggers.get("wandb"):
            self.loggers["wandb"].finish()
        if self.loggers.get("jsonl_handler"):
            self.loggers["jsonl_handler"].close()


def log_gpu_memory(logger: logging.Logger, prefix: str = "system") -> None:
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        logger.info(
            "GPU Memory",
            extra={
                "metrics": {
                    f"{prefix}/gpu_allocated_gb": allocated,
                    f"{prefix}/gpu_reserved_gb": reserved,
                    f"{prefix}/gpu_max_allocated_gb": max_allocated,
                }
            }
        )


def log_system_stats(logger: logging.Logger, prefix: str = "system") -> None:
    """Log system stats (CPU, RAM)."""
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()

        logger.info(
            "System Stats",
            extra={
                "metrics": {
                    f"{prefix}/cpu_percent": cpu_percent,
                    f"{prefix}/ram_percent": ram.percent,
                    f"{prefix}/ram_used_gb": ram.used / 1024**3,
                    f"{prefix}/ram_available_gb": ram.available / 1024**3,
                }
            }
        )
    except ImportError:
        pass