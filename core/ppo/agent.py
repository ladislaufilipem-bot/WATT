# PPO Agent - Main Class
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import numpy as np

from core.neural.actor import create_actor, GaussianActor, ActorOutput
from core.neural.critic import create_critic, Critic, CriticOutput
from core.neural.encoders import create_encoder
from core.ppo.buffer import RolloutBuffer, create_buffer
from core.ppo.update import PPOTrainer, UpdateStats
from core.utils.config import load_config_with_inheritance, apply_cli_overrides
from core.utils.seeding import set_global_seed, get_global_seed_state, set_global_seed_state
from core.utils.logging import setup_logging, MetricsLogger, get_logger
from core.utils.checkpoint import CheckpointManager, save_checkpoint, load_checkpoint
from core.utils.metrics import compute_gae, evaluate_policy, EpisodeStats, MetricsBuffer


@dataclass
class TrainingState:
    """Training state for checkpointing."""
    step: int = 0
    episode: int = 0
    total_timesteps: int = 0
    best_eval_reward: float = float('-inf')
    start_time: float = field(default_factory=time.time)
    rng_states: Dict[str, Any] = field(default_factory=dict)


class PPOAgent:
    """
    Main PPO Agent class.

    Handles training, evaluation, and checkpointing.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        obs_space: Dict[str, Tuple[int, ...]],
        action_space: Tuple[int, ...],
        device: str = "auto",
        seed: int = 42
    ):
        self.config = config
        self.obs_space = obs_space
        self.action_space = action_space
        self.seed = seed

        # Device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Set seed
        set_global_seed(seed, deterministic=config.get("training", {}).get("deterministic", True))

        # Extract config sections
        self.ppo_config = config.get("ppo", {})
        self.neural_config = config.get("neural", {})
        self.env_config = config.get("env", {})
        self.training_config = config.get("training", {})
        self.preprocessing_config = config.get("preprocessing", {})

        # Build networks
        self._build_networks()

        # Build buffer
        self.buffer = create_buffer(
            config=self.ppo_config,
            obs_space=obs_space,
            action_space=action_space,
            n_envs=self.env_config.get("n_envs", 1),
            device=self.device
        )

        # Create trainer
        self.trainer = PPOTrainer(
            actor=self.actor,
            critic=self.critic,
            buffer=self.buffer,
            config=self.ppo_config,
            device=self.device
        )

        # Observation normalization
        self.obs_rms = {}
        if self.preprocessing_config.get("normalize_obs", True):
            for key, shape in obs_space.items():
                self.obs_rms[key] = RunningMeanStd.create(shape)

        # Reward normalization
        self.reward_rms = None
        if self.preprocessing_config.get("normalize_reward", False):
            self.reward_rms = RunningMeanStd.create(())

        # Training state
        self.state = TrainingState()

        # Logging
        self.logger = get_logger(f"ppo_agent")
        self.metrics_logger = None

        # Checkpoint manager
        self.checkpoint_manager = None

        # Callback hooks
        self.callbacks = {
            "on_step": [],
            "on_episode": [],
            "on_epoch": [],
            "on_eval": [],
            "on_save": []
        }

    def _build_networks(self) -> None:
        """Build actor and critic networks."""
        # Determine obs_dim (flattened for MLP, or use first space for encoder)
        # For now, assume we concatenate all observations
        total_obs_dim = sum(np.prod(shape) for shape in self.obs_space.values())

        # Actor
        actor_config = self.neural_config.get("actor", {})
        actor_config["encoder"] = self.neural_config.get("encoders", {}).get("actor", {"type": "mlp"})

        self.actor = create_actor(
            config=actor_config,
            obs_dim=total_obs_dim,
            action_dim=self.action_space[0]
        ).to(self.device)

        # Critic
        critic_config = self.neural_config.get("critic", {})
        critic_config["encoder"] = self.neural_config.get("encoders", {}).get("critic", {"type": "mlp"})

        self.critic = create_critic(
            config=critic_config,
            obs_dim=total_obs_dim,
            action_dim=self.action_space[0]
        ).to(self.device)

        self.logger.info(f"Actor: {sum(p.numel() for p in self.actor.parameters())} params")
        self.logger.info(f"Critic: {sum(p.numel() for p in self.critic.parameters())} params")

    def _flatten_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten dictionary observation to vector."""
        flattened = []
        for key in sorted(obs.keys()):
            if key in self.obs_space:
                x = obs[key]
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                flattened.append(x.flatten(1))
        return torch.cat(flattened, dim=1)

    def _normalize_obs(self, obs: Dict[str, torch.Tensor], update: bool = True) -> Dict[str, torch.Tensor]:
        """Normalize observations."""
        if not self.preprocessing_config.get("normalize_obs", True):
            return obs

        normalized = {}
        clip = self.preprocessing_config.get("obs_clip", 10.0)

        for key, tensor in obs.items():
            if key in self.obs_rms:
                if update and self.training:
                    self.obs_rms[key].update(tensor.cpu().numpy())
                normalized[key] = torch.tensor(
                    self.obs_rms[key].normalize(tensor.cpu().numpy(), clip=clip),
                    device=tensor.device,
                    dtype=tensor.dtype
                )
            else:
                normalized[key] = tensor

        return normalized

    def _normalize_reward(self, reward: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Normalize reward."""
        if self.reward_rms is None:
            return reward

        if update and self.training:
            self.reward_rms.update(reward.cpu().numpy())

        clip = self.preprocessing_config.get("reward_clip", 10.0)
        normalized = self.reward_rms.normalize(reward.cpu().numpy(), clip=clip)
        return torch.tensor(normalized, device=reward.device, dtype=reward.dtype)

    def setup_logging(
        self,
        log_dir: Union[str, Path],
        tensorboard: bool = True,
        wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Setup logging."""
        loggers = setup_logging(
            log_dir=log_dir,
            level=self.training_config.get("log_level", "INFO"),
            console=True,
            jsonl=True,
            tensorboard=tensorboard,
            wandb=wandb,
            wandb_config=wandb_config,
            name=f"agent_{self.config.get('agent', {}).get('name', 'unknown')}"
        )

        self.metrics_logger = MetricsLogger(
            loggers=loggers,
            agent_name=self.config.get("agent", {}).get("name", "agent"),
            log_interval=self.training_config.get("log_interval", 1000)
        )

    def setup_checkpointing(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best: bool = True,
        best_metric: str = "eval/mean_reward"
    ) -> None:
        """Setup checkpoint manager."""
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
            save_best=save_best,
            best_metric=best_metric
        )

    def register_callback(self, event: str, callback) -> None:
        """Register callback for events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Callback {event} failed: {e}")

    def predict(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Predict action for observation.

        Args:
            obs: Dictionary of observations
            deterministic: If True, use deterministic policy

        Returns:
            Action tensor
        """
        self.actor.eval()
        self.critic.eval()

        with torch.no_grad():
            # Normalize
            obs = self._normalize_obs(obs, update=False)

            # Flatten
            obs_flat = self._flatten_obs(obs)

            # Forward
            output = self.actor(obs_flat, deterministic=deterministic)

            # Scale action
            action_scale = self.preprocessing_config.get("action_scale", 1.0)
            action_bias = self.preprocessing_config.get("action_bias", 0.0)
            action = output.action * action_scale + action_bias

        return action

    def predict_batch(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict action with value and log_prob.

        Returns:
            action, value, log_prob
        """
        self.actor.eval()
        self.critic.eval()

        with torch.no_grad():
            obs = self._normalize_obs(obs, update=False)
            obs_flat = self._flatten_obs(obs)

            actor_output = self.actor(obs_flat, deterministic=deterministic)
            critic_output = self.critic(obs_flat)

            action_scale = self.preprocessing_config.get("action_scale", 1.0)
            action_bias = self.preprocessing_config.get("action_bias", 0.0)
            action = actor_output.action * action_scale + action_bias

            return action, critic_output.value, actor_output.log_prob

    def collect_rollout(
        self,
        env,
        n_steps: int,
        n_envs: int = 1
    ) -> Dict[str, float]:
        """
        Collect rollout data.

        Args:
            env: Vectorized environment
            n_steps: Number of steps to collect
            n_envs: Number of environments

        Returns:
            Collection statistics
        """
        self.actor.train()
        self.critic.train()
        self.training = True

        episode_stats = EpisodeStats()
        collection_rewards = []
        collection_lengths = []

        # Reset environments
        obs, _ = env.reset()
        episode_start = True

        for step in range(n_steps):
            # Normalize observations
            obs = self._normalize_obs(obs, update=True)
            obs_flat = self._flatten_obs(obs)

            # Get action, value, log_prob
            with torch.no_grad():
                actor_output = self.actor(obs_flat)
                critic_output = self.critic(obs_flat)

                action = actor_output.action
                value = critic_output.value
                log_prob = actor_output.log_prob

            # Scale action for environment
            action_scale = self.preprocessing_config.get("action_scale", 1.0)
            action_bias = self.preprocessing_config.get("action_bias", 0.0)
            env_action = action * action_scale + action_bias

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(env_action.cpu().numpy())
            done = terminated | truncated

            # Normalize reward
            reward_tensor = torch.tensor(reward, device=self.device, dtype=torch.float32)
            reward_tensor = self._normalize_reward(reward_tensor, update=True)

            # Store in buffer
            self.buffer.add(
                obs=obs,
                action=action,
                reward=reward_tensor,
                value=value,
                log_prob=log_prob,
                done=torch.tensor(done, device=self.device, dtype=torch.bool),
                next_obs=next_obs
            )

            # Episode tracking
            episode_stats.reward += reward.mean() if isinstance(reward, np.ndarray) else reward
            episode_stats.length += 1

            # Callbacks
            self._trigger_callbacks("on_step", step, obs, action, reward, done, info)

            # Handle episode end
            if done.any():
                episode_stats.finalize()
                collection_rewards.append(episode_stats.reward)
                collection_lengths.append(episode_stats.length)

                self._trigger_callbacks("on_episode", episode_stats)

                episode_stats = EpisodeStats()

            obs = next_obs
            self.state.step += 1

            # Log metrics periodically
            if self.metrics_logger and self.state.step % self.training_config.get("log_interval", 1000) == 0:
                self.metrics_logger.log_training_step({
                    "collection/reward_mean": np.mean(collection_rewards) if collection_rewards else 0,
                    "collection/length_mean": np.mean(collection_lengths) if collection_lengths else 0,
                })

        # Get last values for GAE
        with torch.no_grad():
            obs = self._normalize_obs(obs, update=False)
            obs_flat = self._flatten_obs(obs)
            last_values = self.critic(obs_flat).value
            last_dones = torch.tensor(done, device=self.device, dtype=torch.bool)

        return {
            "mean_reward": np.mean(collection_rewards) if collection_rewards else 0,
            "std_reward": np.std(collection_rewards) if collection_rewards else 0,
            "mean_length": np.mean(collection_lengths) if collection_lengths else 0,
            "num_episodes": len(collection_rewards)
        }

    def train(self, env, total_timesteps: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            env: Training environment
            total_timesteps: Override total timesteps from config

        Returns:
            Training history
        """
        total_timesteps = total_timesteps or self.training_config.get("total_timesteps", 1_000_000)
        n_steps = self.ppo_config.get("n_steps", 2048)
        n_envs = self.env_config.get("n_envs", 1)

        self.logger.info(f"Starting training for {total_timesteps} timesteps")
        self.logger.info(f"Rollout length: {n_steps}, Envs: {n_envs}")

        history = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
            "explained_variance": [],
            "entropy": [],
            "learning_rate": [],
            "grad_norm": [],
            "eval_reward": [],
            "eval_length": [],
        }

        eval_freq = self.training_config.get("eval_freq", 50_000)
        save_freq = self.training_config.get("save_freq", 100_000)

        while self.state.total_timesteps < total_timesteps:
            # Collect rollout
            collect_stats = self.collect_rollout(env, n_steps, n_envs)
            self.state.total_timesteps += n_steps * n_envs

            # Train
            last_values = self.buffer.values[-1] if len(self.buffer) > 0 else torch.zeros(n_envs, device=self.device)
            last_dones = self.buffer.dones[-1] if len(self.buffer) > 0 else torch.zeros(n_envs, device=self.device, dtype=torch.bool)

            update_stats_list = self.trainer.train_step(last_values, last_dones)

            # Log update stats
            for stats in update_stats_list:
                history["policy_loss"].append(stats.policy_loss)
                history["value_loss"].append(stats.value_loss)
                history["entropy_loss"].append(stats.entropy_loss)
                history["total_loss"].append(stats.total_loss)
                history["approx_kl"].append(stats.approx_kl)
                history["clip_fraction"].append(stats.clip_fraction)
                history["explained_variance"].append(stats.explained_variance)
                history["entropy"].append(stats.entropy)
                history["learning_rate"].append(stats.learning_rate)
                history["grad_norm"].append(stats.grad_norm)

                if self.metrics_logger:
                    self.metrics_logger.log_training_step({
                        "train/policy_loss": stats.policy_loss,
                        "train/value_loss": stats.value_loss,
                        "train/entropy_loss": stats.entropy_loss,
                        "train/total_loss": stats.total_loss,
                        "train/approx_kl": stats.approx_kl,
                        "train/clip_fraction": stats.clip_fraction,
                        "train/explained_variance": stats.explained_variance,
                        "train/entropy": stats.entropy,
                        "train/learning_rate": stats.learning_rate,
                        "train/grad_norm": stats.grad_norm,
                    })

            # Episode stats
            if self.metrics_logger:
                self.metrics_logger.log_metrics({
                    "collect/mean_reward": collect_stats["mean_reward"],
                    "collect/mean_length": collect_stats["mean_length"],
                    "collect/num_episodes": collect_stats["num_episodes"],
                })

            # Evaluation
            if self.state.total_timesteps % eval_freq == 0:
                eval_stats = self.evaluate(env)
                history["eval_reward"].append(eval_stats["mean_reward"])
                history["eval_length"].append(eval_stats["mean_length"])

                if self.metrics_logger:
                    self.metrics_logger.log_eval({
                        "eval/mean_reward": eval_stats["mean_reward"],
                        "eval/std_reward": eval_stats["std_reward"],
                        "eval/success_rate": eval_stats.get("success_rate", 0),
                        "eval/mean_length": eval_stats["mean_length"],
                    })

                # Save best
                if self.checkpoint_manager and eval_stats["mean_reward"] > self.state.best_eval_reward:
                    self.state.best_eval_reward = eval_stats["mean_reward"]
                    self.save_checkpoint(is_best=True)

            # Periodic save
            if self.checkpoint_manager and self.state.total_timesteps % save_freq == 0:
                self.save_checkpoint(is_latest=True)

            # Callbacks
            self._trigger_callbacks("on_epoch", self.state.total_timesteps, update_stats_list[-1])

        # Final save
        if self.checkpoint_manager:
            self.save_checkpoint(is_latest=True)

        self.logger.info("Training completed")
        return history

    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, float]:
        """Evaluate agent."""
        self.actor.eval()
        self.critic.eval()
        self.training = False

        def policy_fn(obs):
            return self.predict(obs, deterministic=deterministic).cpu().numpy()

        # Use the utility function
        return evaluate_policy(
            policy=policy_fn,
            env=env,
            n_episodes=n_episodes,
            deterministic=deterministic,
            render=render
        )

    def save_checkpoint(
        self,
        path: Optional[Union[str, Path]] = None,
        is_best: bool = False,
        is_latest: bool = False
    ) -> Optional[Path]:
        """Save checkpoint."""
        if self.checkpoint_manager is None and path is None:
            self.logger.warning("No checkpoint manager configured")
            return None

        # Prepare RNG states
        rng_states = get_global_seed_state().to_dict()

        # Prepare metrics
        metrics = {
            "train/policy_loss": self.trainer.get_lr() if hasattr(self.trainer, 'get_lr') else 0,
        }

        if self.checkpoint_manager:
            return self.checkpoint_manager.save(
                model=self.actor,  # Save actor as main model
                optimizer=self.trainer.optimizer,
                scheduler=self.trainer.scheduler,
                step=self.state.step,
                episode=self.state.episode,
                metrics=metrics,
                config=self.config,
                rng_states=rng_states,
                is_best=is_best,
                is_latest=is_latest
            )
        else:
            save_checkpoint(
                path=path,
                model=self.actor,
                optimizer=self.trainer.optimizer,
                scheduler=self.trainer.scheduler,
                step=self.state.step,
                episode=self.state.episode,
                metrics=metrics,
                config=self.config,
                rng_states=rng_states
            )
            return Path(path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load checkpoint."""
        if self.checkpoint_manager:
            metadata = self.checkpoint_manager.load(
                path=path,
                model=self.actor,
                optimizer=self.trainer.optimizer,
                scheduler=self.trainer.scheduler,
                device=self.device
            )
        else:
            result = load_checkpoint(
                path=path,
                model=self.actor,
                optimizer=self.trainer.optimizer,
                scheduler=self.trainer.scheduler,
                device=self.device
            )
            metadata = result["metadata"]

        # Restore state
        self.state.step = metadata.step
        self.state.episode = metadata.episode

        if metadata.rng_states:
            set_global_seed_state(SeedState.from_dict(metadata.rng_states))

        self.logger.info(f"Loaded checkpoint from {path} (step={metadata.step})")

    def get_action_distribution(self, obs: Dict[str, torch.Tensor]) -> ActorOutput:
        """Get full action distribution for observation."""
        self.actor.eval()
        with torch.no_grad():
            obs = self._normalize_obs(obs, update=False)
            obs_flat = self._flatten_obs(obs)
            return self.actor(obs_flat)

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get value estimate for observation."""
        self.critic.eval()
        with torch.no_grad():
            obs = self._normalize_obs(obs, update=False)
            obs_flat = self._flatten_obs(obs)
            return self.critic(obs_flat).value


class RunningMeanStd:
    """Running mean and standard deviation."""

    def __init__(self, shape: tuple, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    @classmethod
    def create(cls, shape: tuple, epsilon: float = 1e-8):
        return cls(shape, epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "var": self.var, "count": np.array(self.count)}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = float(state["count"])


def create_agent_from_config(
    config_path: Union[str, Path],
    obs_space: Dict[str, Tuple[int, ...]],
    action_space: Tuple[int, ...],
    device: str = "auto",
    cli_overrides: Optional[List[str]] = None
) -> PPOAgent:
    """Factory function to create agent from config file."""
    config = load_config_with_inheritance(config_path)

    if cli_overrides:
        config = apply_cli_overrides(config, cli_overrides)

    return PPOAgent(
        config=config,
        obs_space=obs_space,
        action_space=action_space,
        device=device,
        seed=config.get("training", {}).get("seed", 42)
    )