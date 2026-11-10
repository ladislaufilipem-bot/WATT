#!/usr/bin/env python3
# Evaluate Agents - Single or Comparison

import sys
import argparse
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import gymnasium as gym

from core.ppo.agent import create_agent_from_config
from core.env.webots import make_webots_env, WebotsConfig
from core.utils.config import load_yaml, load_config_with_inheritance
from core.utils.seeding import set_global_seed
from core.utils.logging import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO agents")
    parser.add_argument("--agents", type=str, required=True, help="Comma-separated agent names")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Checkpoint file (per agent)")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per agent")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Deterministic policy")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--compare", action="store_true", help="Generate comparison report")
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


def evaluate_agent(agent_name: str, checkpoint: str, episodes: int, deterministic: bool,
                   render: bool, device: str, seed: int) -> dict:
    """Evaluate a single agent."""
    agent_dir = PROJECT_ROOT / "agents" / agent_name
    config_path = agent_dir / "config.yaml"
    checkpoint_path = agent_dir / "checkpoints" / checkpoint

    if not config_path.exists():
        return {"error": f"Config not found: {config_path}"}

    if not checkpoint_path.exists():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    # Load config
    config = load_config_with_inheritance(config_path)

    # Set seed
    set_global_seed(seed, deterministic=True)

    # Create environment
    env = create_env(config, render=render)

    # Create agent
    obs_space = env.observation_space.spaces if hasattr(env.observation_space, 'spaces') else {"obs": env.observation_space}
    action_space = env.action_space.spaces if hasattr(env.action_space, 'spaces') else {"action": env.action_space}

    total_obs_dim = sum(np.prod(s.shape) for s in obs_space.values())
    total_action_dim = sum(np.prod(s.shape) for s in action_space.values())

    agent = create_agent_from_config(
        config_path=config_path,
        obs_space={k: v.shape for k, v in obs_space.items()},
        action_space=(total_action_dim,),
        device=device
    )

    # Load checkpoint
    agent.load_checkpoint(checkpoint_path)

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    successes = 0
    reward_breakdowns = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        length = 0
        ep_breakdown = {}

        while not done:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated or truncated
            total_reward += reward
            length += 1

            # Collect reward breakdown
            if "reward_breakdown" in info:
                for k, v in info["reward_breakdown"].items():
                    ep_breakdown[k] = ep_breakdown.get(k, 0) + v

        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        reward_breakdowns.append(ep_breakdown)

        if info.get("success", False):
            successes += 1

    env.close()

    # Average reward breakdown
    avg_breakdown = {}
    if reward_breakdowns:
        for key in reward_breakdowns[0].keys():
            avg_breakdown[key] = np.mean([b.get(key, 0) for b in reward_breakdowns])

    return {
        "agent": agent_name,
        "checkpoint": checkpoint,
        "episodes": episodes,
        "deterministic": deterministic,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "success_rate": successes / episodes,
        "reward_breakdown": avg_breakdown,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }


def generate_comparison_report(results: list) -> str:
    """Generate markdown comparison report."""
    lines = [
        "# Agent Comparison Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        "| Agent | Mean Reward | Std Reward | Success Rate | Mean Length |",
        "|-------|-------------|------------|--------------|-------------|"
    ]

    for r in results:
        if "error" in r:
            lines.append(f"| {r['agent']} | ERROR: {r['error']} | | | |")
        else:
            lines.append(
                f"| {r['agent']} | {r['mean_reward']:.2f} | "
                f"{r['std_reward']:.2f} | {r['success_rate']:.2%} | {r['mean_length']:.1f} |"
            )

    lines.extend([
        "",
        "## Reward Breakdown",
        ""
    ])

    # Collect all breakdown keys
    all_keys = set()
    for r in results:
        if "reward_breakdown" in r:
            all_keys.update(r["reward_breakdown"].keys())

    if all_keys:
        lines.append("| Agent | " + " | ".join(all_keys) + " |")
        lines.append("|-------|" + "|".join(["------"] * len(all_keys)) + "|")

        for r in results:
            if "error" in r:
                continue
            row = f"| {r['agent']} |"
            for key in all_keys:
                val = r.get("reward_breakdown", {}).get(key, 0)
                row += f" {val:.3f} |"
            lines.append(row)

    return "\n".join(lines)


def main():
    args = parse_args()

    logger = get_logger(__name__)

    agent_names = [a.strip() for a in args.agents.split(",")]

    logger.info(f"Evaluating {len(agent_names)} agents: {agent_names}")

    results = []

    for agent_name in agent_names:
        logger.info(f"Evaluating {agent_name}...")
        result = evaluate_agent(
            agent_name=agent_name,
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            deterministic=args.deterministic,
            render=args.render,
            device=args.device,
            seed=args.seed
        )
        results.append(result)

        if "error" in result:
            logger.error(f"  {agent_name}: {result['error']}")
        else:
            logger.info(f"  {agent_name}: reward={result['mean_reward']:.2f}±{result['std_reward']:.2f}, "
                       f"success={result['success_rate']:.2%}")

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    # Comparison report
    if args.compare:
        from datetime import datetime
        report = generate_comparison_report(results)
        report_path = PROJECT_ROOT / "reports" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        logger.info(f"Comparison report saved to {report_path}")
        print("\n" + report)


if __name__ == "__main__":
    main()