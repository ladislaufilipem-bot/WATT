# Utilities Package
"""
Utility functions for PPO training
"""

from core.utils.config import (
    load_yaml, save_yaml, deep_merge, validate_config,
    load_config_with_inheritance, apply_cli_overrides,
    get_config_value, set_config_value, compute_config_hash,
    ConfigManager
)
from core.utils.seeding import (
    set_global_seed, get_global_seed_state, set_global_seed_state,
    seed_worker, generate_seeds, create_seeded_generators,
    SeededContext, verify_determinism, SeedState
)
from core.utils.logging import (
    setup_logging, get_logger, MetricsLogger, TensorBoardLogger,
    WandbLogger, JSONLHandler, LogRecord
)
from core.utils.metrics import (
    RunningMeanStd, EpisodeStats, MetricsBuffer,
    compute_gae, compute_returns, explained_variance,
    compute_kl_divergence, compute_clip_fraction, compute_entropy,
    gaussian_log_prob, gaussian_entropy, evaluate_policy,
    safe_mean, safe_std
)
from core.utils.checkpoint import (
    CheckpointManager, CheckpointMetadata,
    save_checkpoint, load_checkpoint, ModelExporter
)

__all__ = [
    # Config
    "load_yaml", "save_yaml", "deep_merge", "validate_config",
    "load_config_with_inheritance", "apply_cli_overrides",
    "get_config_value", "set_config_value", "compute_config_hash",
    "ConfigManager",
    # Seeding
    "set_global_seed", "get_global_seed_state", "set_global_seed_state",
    "seed_worker", "generate_seeds", "create_seeded_generators",
    "SeededContext", "verify_determinism", "SeedState",
    # Logging
    "setup_logging", "get_logger", "MetricsLogger", "TensorBoardLogger",
    "WandbLogger", "JSONLHandler", "LogRecord",
    # Metrics
    "RunningMeanStd", "EpisodeStats", "MetricsBuffer",
    "compute_gae", "compute_returns", "explained_variance",
    "compute_kl_divergence", "compute_clip_fraction", "compute_entropy",
    "gaussian_log_prob", "gaussian_entropy", "evaluate_policy",
    "safe_mean", "safe_std",
    # Checkpoint
    "CheckpointManager", "CheckpointMetadata",
    "save_checkpoint", "load_checkpoint", "ModelExporter",
]