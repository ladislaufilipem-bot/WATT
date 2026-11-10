# Global Seeding for Reproducibility
from __future__ import annotations

import os
import random
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

import torch


@dataclass
class SeedState:
    """Container for all RNG states."""
    python_state: tuple
    numpy_state: dict
    torch_state: torch.Tensor
    torch_cuda_state: Optional[torch.Tensor] = None
    torch_mps_state: Optional[torch.Tensor] = None

    def to_dict(self) -> dict:
        return {
            "python": self.python_state,
            "numpy": self.numpy_state,
            "torch": self.torch_state.tolist(),
            "torch_cuda": self.torch_cuda_state.tolist() if self.torch_cuda_state is not None else None,
            "torch_mps": self.torch_mps_state.tolist() if self.torch_mps_state is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SeedState':
        return cls(
            python_state=tuple(data["python"]),
            numpy_state=data["numpy"],
            torch_state=torch.tensor(data["torch"]),
            torch_cuda_state=torch.tensor(data["torch_cuda"]) if data["torch_cuda"] else None,
            torch_mps_state=torch.tensor(data["torch_mps"]) if data["torch_mps"] else None,
        )


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value
        deterministic: If True, enable deterministic algorithms (slower)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # MPS (Apple Silicon)
    if hasattr(torch, 'mps') and torch.mps.is_available():
        torch.mps.manual_seed(seed)

    # Environment variable for other libraries
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Deterministic algorithms
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # CuDNN settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_global_seed_state() -> SeedState:
    """Capture current state of all RNGs."""
    return SeedState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_state=torch.get_rng_state(),
        torch_cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        torch_mps_state=torch.mps.get_rng_state() if hasattr(torch, 'mps') and torch.mps.is_available() else None,
    )


def set_global_seed_state(state: SeedState) -> None:
    """Restore RNG states."""
    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_state)

    if state.torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda_state)

    if state.torch_mps_state is not None and hasattr(torch, 'mps') and torch.mps.is_available():
        torch.mps.set_rng_state(state.torch_mps_state)


def seed_worker(worker_id: int, base_seed: int) -> None:
    """
    Worker init function for DataLoader.
    Each worker gets a unique seed derived from base_seed.
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def generate_seeds(base_seed: int, count: int) -> List[int]:
    """Generate a list of unique seeds from base seed."""
    rng = np.random.RandomState(base_seed)
    return rng.randint(0, 2**31 - 1, size=count, dtype=np.int64).tolist()


def create_seeded_generators(
    base_seed: int,
    num_generators: int,
    device: str = "cpu"
) -> List[torch.Generator]:
    """Create multiple independent torch Generators."""
    seeds = generate_seeds(base_seed, num_generators)
    generators = []

    for seed in seeds:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        generators.append(gen)

    return generators


class SeededContext:
    """
    Context manager for temporary seeding.

    Usage:
        with SeededContext(42):
            # All random ops use seed 42
            x = torch.randn(10)

        # Original RNG state restored
    """

    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.previous_state: Optional[SeedState] = None

    def __enter__(self) -> 'SeededContext':
        self.previous_state = get_global_seed_state()
        set_global_seed(self.seed, self.deterministic)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.previous_state:
            set_global_seed_state(self.previous_state)


def verify_determinism(
    func,
    seed: int = 42,
    num_runs: int = 3,
    *args,
    **kwargs
) -> bool:
    """
    Verify that a function produces deterministic results.

    Returns True if all runs produce identical outputs.
    """
    outputs = []

    for _ in range(num_runs):
        with SeededContext(seed):
            output = func(*args, **kwargs)

        if isinstance(output, torch.Tensor):
            outputs.append(output.detach().cpu().numpy())
        elif isinstance(output, np.ndarray):
            outputs.append(output.copy())
        else:
            outputs.append(output)

    # Compare all outputs
    first = outputs[0]
    for out in outputs[1:]:
        if isinstance(first, np.ndarray):
            if not np.allclose(first, out):
                return False
        elif isinstance(first, (list, tuple)):
            if not all(np.allclose(a, b) for a, b in zip(first, out)):
                return False
        else:
            if first != out:
                return False

    return True


# Auto-seed on import if SEED env var is set
if 'SEED' in os.environ:
    try:
        seed_value = int(os.environ['SEED'])
        set_global_seed(seed_value)
    except ValueError:
        pass