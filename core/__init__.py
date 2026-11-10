# Core Package
"""
PPO Webots - Core Library

Shared components for PPO training with Webots simulation.
"""

from core.ppo.agent import PPOAgent, create_agent_from_config
from core.ppo.buffer import RolloutBuffer, create_buffer
from core.ppo.loss import ppo_loss, compute_gae, PPOLosses
from core.ppo.update import PPOTrainer, PPOUpdater, create_optimizer
from core.neural.actor import create_actor, GaussianActor, ActorOutput
from core.neural.critic import create_critic, Critic, CriticOutput
from core.neural.encoders import create_encoder, MLPEncoder, CNNEncoder, RNNEncoder, LidarEncoder
from core.env.base import BaseEnvWrapper, make_env
from core.env.webots import WebotsEnv, WebotsVecEnv, make_webots_env, WebotsConfig
from core.utils.config import load_config_with_inheritance, validate_config, ConfigManager
from core.utils.seeding import set_global_seed, get_global_seed_state, set_global_seed_state
from core.utils.logging import setup_logging, MetricsLogger, TensorBoardLogger
from core.utils.metrics import RunningMeanStd, EpisodeStats, MetricsBuffer, evaluate_policy
from core.utils.checkpoint import CheckpointManager, save_checkpoint, load_checkpoint, ModelExporter

__version__ = "0.1.0"

__all__ = [
    # PPO
    "PPOAgent", "create_agent_from_config",
    "RolloutBuffer", "create_buffer",
    "ppo_loss", "compute_gae", "PPOLosses",
    "PPOTrainer", "PPOUpdater", "create_optimizer",
    # Neural
    "create_actor", "GaussianActor", "ActorOutput",
    "create_critic", "Critic", "CriticOutput",
    "create_encoder", "MLPEncoder", "CNNEncoder", "RNNEncoder", "LidarEncoder",
    # Environment
    "BaseEnvWrapper", "make_env",
    "WebotsEnv", "WebotsVecEnv", "make_webots_env", "WebotsConfig",
    # Utils
    "load_config_with_inheritance", "validate_config", "ConfigManager",
    "set_global_seed", "get_global_seed_state", "set_global_seed_state",
    "setup_logging", "MetricsLogger", "TensorBoardLogger",
    "RunningMeanStd", "EpisodeStats", "MetricsBuffer", "evaluate_policy",
    "CheckpointManager", "save_checkpoint", "load_checkpoint", "ModelExporter",
]