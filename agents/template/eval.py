#!/usr/bin/env python3
# Template Evaluation Script

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import gymnasium as gym

from core.ppo.agent import create_agent_from_config
from core.env.webots import make_webots_env, WebotsConfig
from core.utils.config import load_config_with_inheritance, apply_cli_overrides
from core.utils.seeding import set_global_seed
from core.utils.logging import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO Agent")
    parser.add_argument("--agent", type=str, required=True, help="Agent directory name")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Checkpoint file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Deterministic policy")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--record", action="store_true", help="Record video")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    return parser.parse_args()


def create_env(config, render: bool = False):
    """Create evaluation environment."""
    env_config = config.get("env", {})

    webots_config = WebotsConfig(
        world=env_config.get("world", "evaluation.wbt"),
        robot_name=env_config.get("robot_name", "ppo_robot"),
        controller=env_config.get("controller", "ppo_agent"),
        rendering=render,
        render_mode="realtime" if render else "fast",
        basic_time_step=env_config.get("basic_time_step", 16),
        max_episode_steps=env_config.get("max_episode_steps", 1000),
        observation_space=env_config.get("observation_space"),
        action_space=env_config.get("action_space")
    )

    return make_webots_env(
        config=webots_config,
        n_envs=1,
        obs_keys=env_config.get("observation_keys"),
        reward_weights=env_config.get("reward_weights")
    )


def main():
    args = parse_args()

    # Paths
    agent_dir = PROJECT_ROOT / "agents" / args.agent
    config_path = agent_dir / "config.yaml"
    checkpoint_path = agent_dir / "checkpoints" / args.checkpoint

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load config
    config = load_config_with_inheritance(config_path)

    # Set seed
    seed = args.seed or config.get("training", {}).get("seed", 42)
    set_global_seed(seed, deterministic=True)

    # Logger
    logger = get_logger(__name__)
    logger.info(f"Evaluating agent: {args.agent}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Deterministic: {args.deterministic}")

    # Create environment
    env = create_env(config, render=args.render)

    # Create agent
    obs_space = env.observation_space.spaces if hasattr(env.observation_space, 'spaces') else {"obs": env.observation_space}
    action_space = env.action_space.spaces if hasattr(env.action_space, 'spaces') else {"action": env.action_space}

    total_obs_dim = sum(np.prod(s.shape) for s in obs_space.values())
    total_action_dim = sum(np.prod(s.shape) for s in action_space.values())

    agent = create_agent_from_config(
        config_path=config_path,
        obs_space={k: v.shape for k, v in obs_space.items()},
        action_space=(total_action_dim,),
        device=args.device
    )

    # Load checkpoint
    logger.info("Loading checkpoint...")
    agent.load_checkpoint(checkpoint_path)

    # Evaluate
    logger.info("Starting evaluation...")

    episode_rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        length = 0

        while not done:
            action = agent.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated or truncated
            total_reward += reward
            length += 1

            if args.record and episode == 0:
                # Record first episode
                pass  # Video recording would go here

        episode_rewards.append(total_reward)
        episode_lengths.append(length)

        if info.get("success", False):
            successes += 1

        logger.info(f"Episode {episode + 1}: reward={total_reward:.2f}, length={length}, success={info.get('success', False)}")

    # Statistics
    results = {
        "agent": args.agent,
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "deterministic": args.deterministic,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "success_rate": successes / args.episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }

    logger.info(f"Results: {results}")

    # Save results
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    env.close()


if __name__ == "__main__":
    main()