# Training Orchestration for Multi-Agent
from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.ppo.agent import PPOAgent, create_agent_from_config
from core.env.webots import make_webots_env, WebotsConfig, WebotsVecEnv
from core.utils.config import load_config_with_inheritance, load_yaml
from core.utils.seeding import set_global_seed
from core.utils.logging import get_logger, setup_logging, MetricsLogger
from core.utils.checkpoint import CheckpointManager
from core.utils.metrics import evaluate_policy


@dataclass
class AgentTrainingState:
    """State of a single agent's training."""
    agent_name: str
    config: Dict[str, Any]
    agent: Optional[PPOAgent] = None
    env: Optional[Any] = None
    metrics_logger: Optional[MetricsLogger] = None
    checkpoint_manager: Optional[CheckpointManager] = None

    # Training progress
    total_timesteps: int = 0
    episode: int = 0
    best_eval_reward: float = float('-inf')
    last_update_time: float = field(default_factory=time.time)

    # Status
    status: str = "initialized"  # initialized, training, evaluating, paused, completed, failed
    error: Optional[str] = None


class TrainingOrchestrator:
    """
    Orchestrates training of multiple agents with optional curriculum,
    population-based training, and resource management.
    """

    def __init__(self, config: Dict[str, Any], base_dir: Path):
        self.config = config
        self.base_dir = Path(base_dir)
        self.agents_dir = self.base_dir / "agents"
        self.config_dir = self.base_dir / "config"

        self.logger = get_logger("orchestrator")

        # Training config
        self.training_config = config.get("training", {})
        self.orchestration_config = config.get("orchestration", {})
        self.resources_config = config.get("resources", {})
        self.monitoring_config = config.get("monitoring", {})
        self.checkpointing_config = config.get("checkpointing", {})
        self.early_stopping_config = config.get("early_stopping", {})

        # Agent states
        self.agent_states: Dict[str, AgentTrainingState] = {}

        # Resource management
        self.max_parallel = self.orchestration_config.get("max_parallel_agents", 2)
        self.gpu_ids = self._get_available_gpus()
        self.current_gpu = 0

        # Population-based training
        self.population_enabled = self.orchestration_config.get("population", {}).get("enabled", False)
        self.population_size = self.orchestration_config.get("population", {}).get("population_size", 10)
        self.perturb_interval = self.orchestration_config.get("population", {}).get("perturb_interval", 100_000)

        # Curriculum
        self.curriculum_enabled = self.orchestration_config.get("curriculum", {}).get("enabled", False)
        self.curriculum_stages = self.orchestration_config.get("curriculum", {}).get("stages", [])
        self.current_stage = 0

        # Global logging
        self.global_logger = None
        self.global_metrics = None

        # Thread pool for parallel training
        self.executor = None

    def _get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs."""
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))

    def setup_global_logging(self) -> None:
        """Setup global logging for orchestrator."""
        log_dir = self.base_dir / "logs" / "orchestrator"
        log_dir.mkdir(parents=True, exist_ok=True)

        loggers = setup_logging(
            log_dir=log_dir,
            level=self.monitoring_config.get("console", {}).get("level", "INFO"),
            console=True,
            jsonl=True,
            tensorboard=self.monitoring_config.get("tensorboard", {}).get("enabled", True),
            wandb=self.monitoring_config.get("wandb", {}).get("enabled", False),
            name="orchestrator"
        )

        self.global_logger = loggers["root"]
        self.global_metrics = MetricsLogger(
            loggers=loggers,
            agent_name="orchestrator",
            log_interval=1000
        )

    def load_agents(self, agent_names: Optional[List[str]] = None) -> None:
        """Load agent configurations and create agents."""
        if agent_names is None:
            agent_names = self.orchestration_config.get("agents", [])

        if not agent_names:
            # Auto-discover agents
            agent_names = [
                d.name for d in self.agents_dir.iterdir()
                if d.is_dir() and d.name != "template" and (d / "config.yaml").exists()
            ]

        self.logger.info(f"Loading {len(agent_names)} agents: {agent_names}")

        for name in agent_names:
            agent_dir = self.agents_dir / name
            config_path = agent_dir / "config.yaml"

            if not config_path.exists():
                self.logger.warning(f"Config not found for {name}, skipping")
                continue

            try:
                config = load_config_with_inheritance(config_path)
                self.agent_states[name] = AgentTrainingState(
                    agent_name=name,
                    config=config
                )
                self.logger.info(f"  Loaded {name}")
            except Exception as e:
                self.logger.error(f"Failed to load {name}: {e}")

    def initialize_agent(self, name: str, gpu_id: Optional[int] = None) -> bool:
        """Initialize a single agent (create env, agent, loggers)."""
        state = self.agent_states[name]

        try:
            config = state.config

            # Set device
            device = "cuda" if gpu_id is not None and torch.cuda.is_available() else "cpu"
            if gpu_id is not None:
                device = f"cuda:{gpu_id}"

            # Create environment
            env_config = config.get("env", {})
            webots_config = WebotsConfig(
                world=env_config.get("world", "training.wbt"),
                robot_name=env_config.get("robot_name", "ppo_robot"),
                controller=env_config.get("controller", "ppo_agent"),
                rendering=False,
                render_mode="fast",
                basic_time_step=env_config.get("basic_time_step", 16),
                max_episode_steps=env_config.get("max_episode_steps", 1000),
                observation_space=env_config.get("observation_space"),
                action_space=env_config.get("action_space"),
                domain_randomization=env_config.get("domain_randomization", False)
            )

            n_envs = config.get("training", {}).get("n_envs", 1)
            state.env = make_webots_env(
                config=webots_config,
                n_envs=n_envs,
                obs_keys=env_config.get("observation_keys"),
                reward_weights=env_config.get("reward_weights")
            )

            # Create agent
            obs_space = state.env.observation_space.spaces if hasattr(state.env.observation_space, 'spaces') else {"obs": state.env.observation_space}
            action_space = state.env.action_space.spaces if hasattr(state.env.action_space, 'spaces') else {"action": state.env.action_space}

            total_obs_dim = sum(np.prod(s.shape) for s in obs_space.values())
            total_action_dim = sum(np.prod(s.shape) for s in action_space.values())

            state.agent = create_agent_from_config(
                config_path=self.agents_dir / name / "config.yaml",
                obs_space={k: v.shape for k, v in obs_space.items()},
                action_space=(total_action_dim,),
                device=device
            )

            # Setup logging
            log_dir = self.agents_dir / name / "logs"
            loggers = setup_logging(
                log_dir=log_dir,
                level=config.get("training", {}).get("log_level", "INFO"),
                console=False,  # No console for parallel agents
                jsonl=True,
                tensorboard=self.monitoring_config.get("tensorboard", {}).get("enabled", True),
                wandb=self.monitoring_config.get("wandb", {}).get("enabled", False),
                name=f"agent_{name}"
            )

            state.metrics_logger = MetricsLogger(
                loggers=loggers,
                agent_name=name,
                log_interval=config.get("training", {}).get("log_interval", 1000)
            )

            # Setup checkpointing
            checkpoint_dir = self.agents_dir / name / "checkpoints"
            state.checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                max_checkpoints=self.checkpointing_config.get("keep_periodic", 5),
                save_best=self.checkpointing_config.get("save_best", True),
                best_metric=self.checkpointing_config.get("best_metric", "eval/mean_reward"),
                best_mode="max" if self.checkpointing_config.get("best_mode", "max") == "max" else "min"
            )

            state.status = "ready"
            return True

        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            self.logger.error(f"Failed to initialize {name}: {e}")
            return False

    def train_agent(self, name: str, total_timesteps: Optional[int] = None) -> Dict[str, Any]:
        """Train a single agent."""
        state = self.agent_states[name]

        if state.status != "ready":
            self.initialize_agent(name)

        if state.status != "ready":
            return {"success": False, "error": state.error}

        config = state.config
        agent = state.agent
        env = state.env
        metrics_logger = state.metrics_logger
        checkpoint_manager = state.checkpoint_manager

        total_timesteps = total_timesteps or config.get("training", {}).get("total_timesteps", 1_000_000)
        n_steps = config.get("ppo", {}).get("n_steps", 2048)
        n_envs = config.get("training", {}).get("n_envs", 1)
        eval_freq = config.get("training", {}).get("eval_freq", 50_000)
        save_freq = config.get("training", {}).get("save_freq", 100_000)
        n_eval_episodes = config.get("training", {}).get("n_eval_episodes", 10)

        state.status = "training"
        state.last_update_time = time.time()

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

        try:
            while state.total_timesteps < total_timesteps:
                # Check for curriculum stage change
                if self.curriculum_enabled:
                    self._check_curriculum_stage(state)

                # Collect rollout
                collect_stats = agent.collect_rollout(env, n_steps, n_envs)
                state.total_timesteps += n_steps * n_envs

                # Train
                last_values = agent.buffer.values[-1] if len(agent.buffer) > 0 else torch.zeros(n_envs, device=agent.device)
                last_dones = agent.buffer.dones[-1] if len(agent.buffer) > 0 else torch.zeros(n_envs, device=agent.device, dtype=torch.bool)

                update_stats_list = agent.trainer.train_step(last_values, last_dones)

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

                    if metrics_logger:
                        metrics_logger.log_training_step({
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

                # Log collection stats
                if metrics_logger:
                    metrics_logger.log_metrics({
                        "collect/mean_reward": collect_stats["mean_reward"],
                        "collect/mean_length": collect_stats["mean_length"],
                        "collect/num_episodes": collect_stats["num_episodes"],
                    })

                # Evaluation
                if state.total_timesteps % eval_freq == 0:
                    state.status = "evaluating"
                    eval_stats = agent.evaluate(env, n_episodes=n_eval_episodes, deterministic=True)
                    state.status = "training"

                    history["eval_reward"].append(eval_stats["mean_reward"])
                    history["eval_length"].append(eval_stats["mean_length"])

                    if metrics_logger:
                        metrics_logger.log_eval({
                            "eval/mean_reward": eval_stats["mean_reward"],
                            "eval/std_reward": eval_stats["std_reward"],
                            "eval/success_rate": eval_stats.get("success_rate", 0),
                            "eval/mean_length": eval_stats["mean_length"],
                        })

                    # Save best
                    if checkpoint_manager and eval_stats["mean_reward"] > state.best_eval_reward:
                        state.best_eval_reward = eval_stats["mean_reward"]
                        checkpoint_manager.save(
                            model=agent.actor,
                            optimizer=agent.trainer.optimizer,
                            scheduler=agent.trainer.scheduler,
                            step=state.total_timesteps,
                            episode=state.episode,
                            metrics={"eval/mean_reward": eval_stats["mean_reward"]},
                            config=config,
                            is_best=True
                        )

                # Periodic save
                if checkpoint_manager and state.total_timesteps % save_freq == 0:
                    checkpoint_manager.save(
                        model=agent.actor,
                        optimizer=agent.trainer.optimizer,
                        scheduler=agent.trainer.scheduler,
                        step=state.total_timesteps,
                        episode=state.episode,
                        metrics={},
                        config=config,
                        is_latest=True
                    )

                # Population-based training perturbation
                if self.population_enabled and state.total_timesteps % self.perturb_interval == 0:
                    self._perturb_agent(state)

                state.last_update_time = time.time()

            # Final save
            if checkpoint_manager:
                checkpoint_manager.save(
                    model=agent.actor,
                    optimizer=agent.trainer.optimizer,
                    scheduler=agent.trainer.scheduler,
                    step=state.total_timesteps,
                    episode=state.episode,
                    metrics={},
                    config=config,
                    is_latest=True
                )

            state.status = "completed"
            return {"success": True, "history": history, "best_eval_reward": state.best_eval_reward}

        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            self.logger.error(f"Agent {name} failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "history": history}

    def _check_curriculum_stage(self, state: AgentTrainingState) -> None:
        """Check and apply curriculum stage changes."""
        for i, stage in enumerate(self.curriculum_stages):
            if state.total_timesteps >= stage.get("timesteps", 0):
                if i > self.current_stage:
                    self.current_stage = i
                    self.logger.info(f"Curriculum stage {i}: {stage['name']}")

                    # Apply config overrides
                    overrides = stage.get("config_overrides", {})
                    self._apply_config_overrides(state, overrides)

    def _apply_config_overrides(self, state: AgentTrainingState, overrides: Dict) -> None:
        """Apply configuration overrides to agent."""
        # This would modify agent config dynamically
        # For now, just log
        self.logger.info(f"Applying overrides to {state.agent_name}: {overrides}")

    def _perturb_agent(self, state: AgentTrainingState) -> None:
        """Perturb agent hyperparameters for population-based training."""
        import random

        perturb_factors = self.orchestration_config.get("population", {}).get("perturb_factors", {})

        for param, (min_factor, max_factor) in perturb_factors.items():
            if hasattr(state.agent.trainer.optimizer.param_groups[0], param):
                current = state.agent.trainer.optimizer.param_groups[0][param]
                factor = random.uniform(min_factor, max_factor)
                new_value = current * factor
                state.agent.trainer.optimizer.param_groups[0][param] = new_value
                self.logger.info(f"Perturbed {state.agent_name}.{param}: {current} -> {new_value}")

    def train_all(
        self,
        agent_names: Optional[List[str]] = None,
        total_timesteps: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Train all agents, respecting parallelism limits."""
        self.setup_global_logging()

        if agent_names is None:
            agent_names = list(self.agent_states.keys())

        # Initialize all agents first
        for name in agent_names:
            gpu_id = self.gpu_ids[len([s for s in self.agent_states.values() if s.status == "ready"]) % len(self.gpu_ids)] if self.gpu_ids else None
            self.initialize_agent(name, gpu_id)

        # Train sequentially with parallel limit
        results = {}

        # For true parallelism, we'd use ThreadPoolExecutor
        # But Webots doesn't support true parallelism well, so sequential with GPU rotation
        for name in agent_names:
            if self.agent_states[name].status == "ready":
                self.logger.info(f"Starting training for {name}")
                results[name] = self.train_agent(name, total_timesteps)
            else:
                results[name] = {"success": False, "error": "Not ready"}

        return results

    def evaluate_all(
        self,
        agent_names: Optional[List[str]] = None,
        n_episodes: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all agents."""
        if agent_names is None:
            agent_names = list(self.agent_states.keys())

        results = {}

        for name in agent_names:
            state = self.agent_states[name]

            if state.agent is None:
                # Load agent for evaluation
                self.initialize_agent(name)

            if state.agent and state.env:
                eval_stats = state.agent.evaluate(state.env, n_episodes=n_episodes, deterministic=True)
                results[name] = eval_stats
                self.logger.info(f"{name} evaluation: {eval_stats}")
            else:
                results[name] = {"error": "Agent not initialized"}

        return results

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents."""
        return {
            name: {
                "status": state.status,
                "total_timesteps": state.total_timesteps,
                "episode": state.episode,
                "best_eval_reward": state.best_eval_reward,
                "error": state.error
            }
            for name, state in self.agent_states.items()
        }


def create_orchestrator(config_path: Union[str, Path], base_dir: Union[str, Path]) -> TrainingOrchestrator:
    """Factory function to create orchestrator from config."""
    config = load_yaml(config_path)
    return TrainingOrchestrator(config, Path(base_dir))