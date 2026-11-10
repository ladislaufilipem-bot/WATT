#!/usr/bin/env python3
# Template Training Script
# Copy this to your agent directory and customize

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import gymnasium as gym
import numpy as np

from core.ppo.agent import create_agent_from_config, PPOAgent
from core.env.webots import make_webots_env, WebotsConfig
from core.utils.config import load_config_with_inheritance, apply_cli_overrides
from core.utils.seeding import set_global_seed
from core.utils.logging import setup_logging, MetricsLogger, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO Agent")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Quick test run (1 episode)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic evaluation")
    parser.add_argument("--render", action="store_true", help="Render evaluation")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--override", action="append", default=[], help="Config overrides (key=value)")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="ppo-webots", help="W&B project")
    return parser.parse_args()


def create_env(config, render: bool = False):
    """Create environment from config."""
    env_config = config.get("env", {})

    webots_config = WebotsConfig(
        world=env_config.get("world", "training.wbt"),
        robot_name=env_config.get("robot_name", "ppo_robot"),
        controller=env_config.get("controller", "ppo_agent"),
        rendering=render,
        render_mode="realtime" if render else "fast",
        basic_time_step=env_config.get("basic_time_step", 16),
        max_episode_steps=env_config.get("max_episode_steps", 1000),
        observation_space=env_config.get("observation_space"),
        action_space=env_config.get("action_space"),
        domain_randomization=env_config.get("domain_randomization", False),
        randomization_config=env_config.get("randomization_config", {})
    )

    return make_webots_env(
        config=webots_config,
        n_envs=config.get("training", {}).get("n_envs", 1),
        obs_keys=env_config.get("observation_keys"),
        reward_weights=env_config.get("reward_weights")
    )


def custom_reward_fn(obs, action, next_obs, info):
    """Custom reward function - OVERRIDE THIS for your task."""
    reward = 0.0
    breakdown = {}

    # Example: Forward velocity reward
    if "joint_vel" in next_obs and len(next_obs["joint_vel"]) > 0:
        forward_vel = next_obs["joint_vel"][0]
        weight = info.get("reward_weights", {}).get("forward_vel", 1.0)
        reward += weight * forward_vel
        breakdown["forward_vel"] = weight * forward_vel

    # Energy penalty
    if action is not None:
        energy = np.sum(np.square(action))
        weight = info.get("reward_weights", {}).get("energy", -0.01)
        reward += weight * energy
        breakdown["energy"] = weight * energy

    # Alive bonus
    alive_weight = info.get("reward_weights", {}).get("alive", 0.1)
    reward += alive_weight
    breakdown["alive"] = alive_weight

    # Task-specific rewards (add your own)
    # if "target_distance" in next_obs:
    #     dist_reward = -next_obs["target_distance"] * 0.1
    #     reward += dist_reward
    #     breakdown["target_distance"] = dist_reward

    info["reward_breakdown"] = breakdown
    return reward, breakdown


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    config = load_config_with_inheritance(config_path)

    # Apply CLI overrides
    if args.override:
        config = apply_cli_overrides(config, args.override)

    # Override seed if provided
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed

    # Set global seed
    seed = config.get("training", {}).get("seed", 42)
    set_global_seed(seed, deterministic=config.get("training", {}).get("deterministic", True))

    # Setup logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    loggers = setup_logging(
        log_dir=log_dir,
        level=config.get("training", {}).get("log_level", "INFO"),
        console=True,
        jsonl=True,
        tensorboard=config.get("logging", {}).get("tensorboard", True),
        wandb=args.wandb,
        wandb_config={
            "project": args.wandb_project,
            "config": config,
            "name": config.get("agent", {}).get("name", "agent")
        } if args.wandb else None,
        name=f"train_{config.get('agent', {}).get('name', 'agent')}"
    )

    logger = get_logger(__name__)
    logger.info(f"Starting training for agent: {config.get('agent', {}).get('name')}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Device: {args.device}")

    # Create environment
    logger.info("Creating environment...")
    env = create_env(config, render=args.render)

    # Create agent
    logger.info("Creating agent...")
    obs_space = env.observation_space.spaces if hasattr(env.observation_space, 'spaces') else {"obs": env.observation_space}
    action_space = env.action_space.spaces if hasattr(env.action_space, 'spaces') else {"action": env.action_space}

    # Flatten spaces for agent
    total_obs_dim = sum(np.prod(s.shape) for s in obs_space.values())
    total_action_dim = sum(np.prod(s.shape) for s in action_space.values())

    agent = create_agent_from_config(
        config_path=config_path,
        obs_space={k: v.shape for k, v in obs_space.items()},
        action_space=(total_action_dim,),
        device=args.device,
        cli_overrides=args.override
    )

    # Set custom reward function
    env_config = config.get("env", {})
    agent.env_config = env_config  # Pass for reward function
    agent.set_reward_function(custom_reward_fn)

    # Setup metrics logger
    metrics_logger = MetricsLogger(
        loggers=loggers,
        agent_name=config.get("agent", {}).get("name", "agent"),
        log_interval=config.get("training", {}).get("log_interval", 1000)
    )
    agent.metrics_logger = metrics_logger

    # Setup checkpointing
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    from core.utils.checkpoint import CheckpointManager
    agent.setup_checkpointing(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=5,
        save_best=True,
        best_metric="eval/mean_reward"
    )

    # Resume if requested
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        agent.load_checkpoint(args.resume)

    # Evaluation only
    if args.eval_only:
        logger.info("Running evaluation...")
        eval_stats = agent.evaluate(env, n_episodes=args.episodes, deterministic=args.deterministic, render=args.render)
        logger.info(f"Evaluation results: {eval_stats}")
        return

    # Dry run
    if args.dry_run:
        logger.info("Running dry run (1 episode)...")
        agent.training = True
        collect_stats = agent.collect_rollout(env, n_steps=min(100, config.get("ppo", {}).get("n_steps", 2048)))
        logger.info(f"Dry run stats: {collect_stats}")
        return

    # Train
    logger.info("Starting training...")
    try:
        history = agent.train(env)
        logger.info("Training completed successfully")

        # Final evaluation
        logger.info("Running final evaluation...")
        eval_stats = agent.evaluate(env, n_episodes=args.episodes, deterministic=True)
        logger.info(f"Final evaluation: {eval_stats}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        agent.save_checkpoint(is_latest=True)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        agent.save_checkpoint(is_latest=True)
        raise

    finally:
        env.close()
        metrics_logger.close()


if __name__ == "__main__":
    main()