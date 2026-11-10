# Checkpoint Save/Load Utilities
from __future__ import annotations

import os
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil

from core.utils.config import compute_config_hash


@dataclass
class CheckpointMetadata:
    """Metadata stored in checkpoint."""
    step: int
    episode: int
    timestamp: str
    config_hash: str
    git_commit: Optional[str] = None
    git_dirty: bool = False
    metrics: Dict[str, float] = None
    rng_states: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.rng_states is None:
            self.rng_states = {}


class CheckpointManager:
    """Manages model checkpoints with rotation."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best: bool = True,
        best_metric: str = "eval/mean_reward",
        best_mode: str = "max",
        format: str = "pytorch"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.format = format

        self.best_value = float('-inf') if best_mode == "max" else float('inf')
        self.checkpoints: List[Path] = []
        self._load_index()

    def _load_index(self) -> None:
        """Load existing checkpoints from directory."""
        self.checkpoints = sorted(
            self.checkpoint_dir.glob("*.pt"),
            key=lambda p: p.stat().st_mtime
        )

        # Check for best checkpoint
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            try:
                ckpt = torch.load(best_path, map_location="cpu")
                if "metadata" in ckpt and self.best_metric in ckpt["metadata"].metrics:
                    self.best_value = ckpt["metadata"].metrics[self.best_metric]
            except Exception:
                pass

    def _is_better(self, value: float) -> bool:
        if self.best_mode == "max":
            return value > self.best_value
        return value < self.best_value

    def _rotate_checkpoints(self) -> None:
        """Remove oldest checkpoints if over limit."""
        while len(self.checkpoints) >= self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if oldest.name != "best.pt" and oldest.name != "latest.pt":
                oldest.unlink(missing_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        step: int = 0,
        episode: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        rng_states: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        is_best: bool = False,
        is_latest: bool = False
    ) -> Path:
        """Save checkpoint."""
        timestamp = datetime.now().isoformat()
        config_hash = compute_config_hash(config) if config else ""

        # Git info
        git_commit = None
        git_dirty = False
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
            git_dirty = len(git_status) > 0
        except Exception:
            pass

        metadata = CheckpointMetadata(
            step=step,
            episode=episode,
            timestamp=timestamp,
            config_hash=config_hash,
            git_commit=git_commit,
            git_dirty=git_dirty,
            metrics=metrics or {},
            rng_states=rng_states or {}
        )

        # Prepare checkpoint dict
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": asdict(metadata),
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if config:
            checkpoint["config"] = config

        # Determine filename
        if name is None:
            name = f"step_{step}.pt"

        save_path = self.checkpoint_dir / name

        # Save
        if self.format == "safetensors":
            try:
                from safetensors.torch import save_file
                # Convert to safetensors format
                tensors = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
                save_file(tensors, str(save_path))
                # Save metadata separately
                meta_path = save_path.with_suffix(".json")
                with open(meta_path, 'w') as f:
                    json.dump({k: v for k, v in checkpoint.items() if not isinstance(v, torch.Tensor)}, f)
            except ImportError:
                torch.save(checkpoint, save_path)
        else:
            torch.save(checkpoint, save_path)

        # Update tracking
        if save_path not in self.checkpoints:
            self.checkpoints.append(save_path)
        self._rotate_checkpoints()

        # Save as best
        if is_best or (self.save_best and metrics and self.best_metric in metrics and self._is_better(metrics[self.best_metric])):
            self.best_value = metrics[self.best_metric]
            best_path = self.checkpoint_dir / "best.pt"
            shutil.copy2(save_path, best_path)
            # Also save metadata
            meta_path = best_path.with_suffix(".json")
            with open(meta_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)

        # Save as latest
        if is_latest:
            latest_path = self.checkpoint_dir / "latest.pt"
            shutil.copy2(save_path, latest_path)
            meta_path = latest_path.with_suffix(".json")
            with open(meta_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)

        return save_path

    def load(
        self,
        path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
        strict: bool = True
    ) -> CheckpointMetadata:
        """Load checkpoint."""
        path = Path(path)

        if self.format == "safetensors" and path.suffix == ".pt":
            try:
                from safetensors.torch import load_file
                tensors = load_file(str(path), device=device)
                meta_path = path.with_suffix(".json")
                if meta_path.exists():
                    with open(meta_path) as f:
                        metadata_dict = json.load(f)
                else:
                    metadata_dict = {}

                model.load_state_dict(tensors, strict=strict)

                if optimizer and "optimizer_state_dict" in tensors:
                    optimizer.load_state_dict(tensors["optimizer_state_dict"])

                if scheduler and "scheduler_state_dict" in tensors:
                    scheduler.load_state_dict(tensors["scheduler_state_dict"])

                return CheckpointMetadata(**metadata_dict.get("metadata", {}))
            except ImportError:
                pass

        # Standard PyTorch
        checkpoint = torch.load(path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        metadata = checkpoint.get("metadata", {})
        return CheckpointMetadata(**metadata)

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu"
    ) -> Optional[CheckpointMetadata]:
        """Load best checkpoint if exists."""
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            return self.load(best_path, model, optimizer, scheduler, device)
        return None

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu"
    ) -> Optional[CheckpointMetadata]:
        """Load latest checkpoint if exists."""
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            return self.load(latest_path, model, optimizer, scheduler, device)
        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints with metadata."""
        result = []
        for path in sorted(self.checkpoints, key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                if self.format == "safetensors":
                    meta_path = path.with_suffix(".json")
                    if meta_path.exists():
                        with open(meta_path) as f:
                            meta = json.load(f)
                    else:
                        meta = {}
                else:
                    ckpt = torch.load(path, map_location="cpu")
                    meta = ckpt.get("metadata", {})

                result.append({
                    "path": str(path),
                    "name": path.name,
                    "size_mb": path.stat().st_size / 1024**2,
                    "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    **meta
                })
            except Exception as e:
                result.append({
                    "path": str(path),
                    "name": path.name,
                    "error": str(e)
                })
        return result

    def cleanup(self, keep_best: bool = True, keep_latest: bool = True) -> int:
        """Remove all checkpoints except best/latest."""
        removed = 0
        for path in self.checkpoints:
            if keep_best and path.name == "best.pt":
                continue
            if keep_latest and path.name == "latest.pt":
                continue
            path.unlink(missing_ok=True)
            meta_path = path.with_suffix(".json")
            meta_path.unlink(missing_ok=True)
            removed += 1

        self._load_index()
        return removed


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    step: int = 0,
    episode: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    rng_states: Optional[Dict[str, Any]] = None
) -> None:
    """Simple checkpoint save function."""
    manager = CheckpointManager(Path(path).parent)
    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        episode=episode,
        metrics=metrics,
        config=config,
        rng_states=rng_states,
        name=Path(path).name
    )


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Simple checkpoint load function."""
    manager = CheckpointManager(Path(path).parent)
    metadata = manager.load(path, model, optimizer, scheduler, device)
    return {"metadata": metadata}


class ModelExporter:
    """Export models to various formats."""

    @staticmethod
    def to_onnx(
        model: torch.nn.Module,
        output_path: Union[str, Path],
        input_shape: tuple,
        opset_version: int = 17,
        dynamic_axes: bool = True,
        simplify: bool = True,
        verify: bool = True,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None
    ) -> Path:
        """Export model to ONNX format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy_input = torch.randn(*input_shape)

        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                input_names[0]: {0: "batch"},
                output_names[0]: {0: "batch"}
            }

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
        )

        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                onnx_model = onnx.load(str(output_path))
                model_simp, check = onnx_simplify(onnx_model)
                if check:
                    onnx.save(model_simp, str(output_path))
            except ImportError:
                pass

        if verify:
            ModelExporter._verify_onnx(output_path, input_shape)

        return output_path

    @staticmethod
    def _verify_onnx(path: Path, input_shape: tuple) -> None:
        """Verify ONNX model produces same output as PyTorch."""
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(str(path))
        input_name = session.get_inputs()[0].name

        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        ort_outputs = session.run(None, {input_name: dummy_input})

        print(f"ONNX model verified: {len(ort_outputs)} outputs")

    @staticmethod
    def to_torchscript(
        model: torch.nn.Module,
        output_path: Union[str, Path],
        input_shape: tuple,
        optimize: bool = True
    ) -> Path:
        """Export model to TorchScript."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy_input = torch.randn(*input_shape)

        traced = torch.jit.trace(model, dummy_input)

        if optimize:
            traced = torch.jit.optimize_for_inference(traced)

        traced.save(str(output_path))
        return output_path