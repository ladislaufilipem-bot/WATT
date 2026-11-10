# Evaluation Module
from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from core.ppo.agent import PPOAgent
from core.utils.logging import get_logger
from core.utils.metrics import evaluate_policy


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    n_episodes: int = 10
    deterministic: bool = True
    render: bool = False
    record_video: bool = False
    video_dir: Optional[str] = None
    seed: int = 42
    metrics: List[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "mean_reward", "std_reward", "min_reward", "max_reward",
                "mean_length", "std_length", "success_rate"
            ]


class AgentEvaluator:
    """Comprehensive agent evaluation."""

    def __init__(self, agent: PPOAgent, env, config: EvaluationConfig = None):
        self.agent = agent
        self.env = env
        self.config = config or EvaluationConfig()
        self.logger = get_logger("evaluator")

    def evaluate(
        self,
        n_episodes: Optional[int] = None,
        deterministic: Optional[bool] = None,
        render: Optional[bool] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run evaluation."""
        n_episodes = n_episodes or self.config.n_episodes
        deterministic = deterministic if deterministic is not None else self.config.deterministic
        render = render or self.config.render

        self.logger.info(f"Evaluating for {n_episodes} episodes (deterministic={deterministic})")

        def policy_fn(obs):
            return self.agent.predict(obs, deterministic=deterministic).cpu().numpy()

        # Use utility function
        results = evaluate_policy(
            policy=policy_fn,
            env=self.env,
            n_episodes=n_episodes,
            deterministic=deterministic,
            render=render,
            callback=callback
        )

        # Add metadata
        results.update({
            "n_episodes": n_episodes,
            "deterministic": deterministic,
            "timestamp": datetime.now().isoformat()
        })

        self.logger.info(f"Evaluation results: {results}")
        return results

    def evaluate_with_breakdown(
        self,
        n_episodes: Optional[int] = None,
        deterministic: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Evaluate with per-component reward breakdown."""
        n_episodes = n_episodes or self.config.n_episodes
        deterministic = deterministic if deterministic is not None else self.config.deterministic

        episode_rewards = []
        episode_lengths = []
        reward_breakdowns = []
        successes = 0

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0.0
            length = 0
            ep_breakdown = {}

            while not done:
                action = self.agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
                done = terminated or truncated
                total_reward += reward
                length += 1

                if "reward_breakdown" in info:
                    for k, v in info["reward_breakdown"].items():
                        ep_breakdown[k] = ep_breakdown.get(k, 0) + v

            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            reward_breakdowns.append(ep_breakdown)

            if info.get("success", False):
                successes += 1

        # Average breakdown
        avg_breakdown = {}
        if reward_breakdowns:
            for key in reward_breakdowns[0].keys():
                avg_breakdown[key] = np.mean([b.get(key, 0) for b in reward_breakdowns])

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "std_length": float(np.std(episode_lengths)),
            "success_rate": successes / n_episodes,
            "reward_breakdown": avg_breakdown,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "n_episodes": n_episodes,
            "deterministic": deterministic,
            "timestamp": datetime.now().isoformat()
        }

    def evaluate_checkpoints(
        self,
        checkpoint_dir: str,
        n_episodes: int = 5,
        pattern: str = "step_*.pt"
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple checkpoints."""
        from core.utils.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir)
        checkpoints = manager.list_checkpoints()

        # Filter checkpoints
        filtered = [c for c in checkpoints if c["name"].startswith("step_")]
        filtered.sort(key=lambda x: x.get("step", 0))

        results = []

        for ckpt in filtered:
            self.logger.info(f"Evaluating checkpoint: {ckpt['name']}")

            # Load checkpoint
            state = manager.load(
                path=ckpt["path"],
                model=self.agent.actor,
                optimizer=self.agent.trainer.optimizer,
                scheduler=self.agent.trainer.scheduler,
                device=self.agent.device
            )

            # Evaluate
            eval_results = self.evaluate(n_episodes=n_episodes)
            eval_results["checkpoint"] = ckpt["name"]
            eval_results["step"] = ckpt.get("step", 0)

            results.append(eval_results)

        return results


def compare_agents(
    agents: Dict[str, PPOAgent],
    env,
    n_episodes: int = 10,
    deterministic: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple agents."""
    results = {}

    for name, agent in agents.items():
        evaluator = AgentEvaluator(agent, env)
        results[name] = evaluator.evaluate(n_episodes=n_episodes, deterministic=deterministic)

    return results


def generate_evaluation_report(
    results: Dict[str, Any],
    output_path: str
) -> None:
    """Generate evaluation report in markdown."""
    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        "| Agent | Mean Reward | Std Reward | Success Rate | Mean Length |",
        "|-------|-------------|------------|--------------|-------------|"
    ]

    for name, result in results.items():
        if "error" in result:
            lines.append(f"| {name} | ERROR: {result['error']} | | | |")
        else:
            lines.append(
                f"| {name} | {result.get('mean_reward', 0):.2f} | "
                f"{result.get('std_reward', 0):.2f} | "
                f"{result.get('success_rate', 0):.2%} | "
                f"{result.get('mean_length', 0):.1f} |"
            )

    lines.extend(["", "## Detailed Results", ""])

    for name, result in results.items():
        lines.append(f"### {name}")
        lines.append("")

        if "error" in result:
            lines.append(f"Error: {result['error']}")
            continue

        lines.append(f"- **Episodes**: {result.get('n_episodes', 'N/A')}")
        lines.append(f"- **Deterministic**: {result.get('deterministic', 'N/A')}")
        lines.append(f"- **Mean Reward**: {result.get('mean_reward', 0):.4f}")
        lines.append(f"- **Std Reward**: {result.get('std_reward', 0):.4f}")
        lines.append(f"- **Min Reward**: {result.get('min_reward', 0):.4f}")
        lines.append(f"- **Max Reward**: {result.get('max_reward', 0):.4f}")
        lines.append(f"- **Mean Length**: {result.get('mean_length', 0):.2f}")
        lines.append(f"- **Std Length**: {result.get('std_length', 0):.2f}")
        lines.append(f"- **Success Rate**: {result.get('success_rate', 0):.2%}")

        if "reward_breakdown" in result:
            lines.append("")
            lines.append("#### Reward Breakdown")
            lines.append("")
            lines.append("| Component | Average |")
            lines.append("|-----------|---------|")
            for comp, val in result["reward_breakdown"].items():
                lines.append(f"| {comp} | {val:.4f} |")

        lines.append("")

    Path(output_path).write_text("\n".join(lines))