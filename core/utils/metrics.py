# Metrics Computation Utilities
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
from dataclasses import dataclass, field
import time


@dataclass
class RunningMeanStd:
    """Running mean and standard deviation (Welford's algorithm)."""
    mean: np.ndarray
    var: np.ndarray
    count: float = 1e-4
    epsilon: float = 1e-8

    @classmethod
    def create(cls, shape: tuple, epsilon: float = 1e-8) -> 'RunningMeanStd':
        return cls(
            mean=np.zeros(shape, dtype=np.float64),
            var=np.ones(shape, dtype=np.float64),
            epsilon=epsilon
        )

    def update(self, x: np.ndarray) -> None:
        """Update with new batch of data."""
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
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
        """Normalize input using running statistics."""
        x = np.asarray(x, dtype=np.float32)
        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize input."""
        return x * np.sqrt(self.var + self.epsilon) + self.mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "var": self.var, "count": np.array(self.count)}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = float(state["count"])


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    reward: float = 0.0
    length: int = 0
    info: Dict[str, float] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def finalize(self) -> None:
        self.end_time = time.time()


class MetricsBuffer:
    """Buffer for computing rolling metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards: deque = deque(maxlen=window_size)
        self.lengths: deque = deque(maxlen=window_size)
        self.infos: deque = deque(maxlen=window_size)
        self.custom: Dict[str, deque] = {}

    def add(self, reward: float, length: int, info: Optional[Dict[str, float]] = None, **custom) -> None:
        self.rewards.append(reward)
        self.lengths.append(length)
        self.infos.append(info or {})

        for key, value in custom.items():
            if key not in self.custom:
                self.custom[key] = deque(maxlen=self.window_size)
            self.custom[key].append(value)

    def get_stats(self) -> Dict[str, float]:
        stats = {}

        if self.rewards:
            rewards_arr = np.array(self.rewards)
            stats.update({
                "mean_reward": float(np.mean(rewards_arr)),
                "std_reward": float(np.std(rewards_arr)),
                "min_reward": float(np.min(rewards_arr)),
                "max_reward": float(np.max(rewards_arr)),
                "median_reward": float(np.median(rewards_arr)),
            })

        if self.lengths:
            lengths_arr = np.array(self.lengths)
            stats.update({
                "mean_length": float(np.mean(lengths_arr)),
                "std_length": float(np.std(lengths_arr)),
            })

        # Aggregate info dicts
        if self.infos:
            all_keys = set()
            for info in self.infos:
                all_keys.update(info.keys())

            for key in all_keys:
                values = [info.get(key, 0.0) for info in self.infos]
                stats[f"mean_{key}"] = float(np.mean(values))

        # Custom metrics
        for key, values in self.custom.items():
            if values:
                stats[f"mean_{key}"] = float(np.mean(values))

        return stats

    def clear(self) -> None:
        self.rewards.clear()
        self.lengths.clear()
        self.infos.clear()
        for d in self.custom.values():
            d.clear()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    next_value: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: [T] or [T, N] - rewards at each step
        values: [T] or [T, N] - value estimates at each step
        dones: [T] or [T, N] - done flags
        gamma: Discount factor
        gae_lambda: GAE lambda
        next_value: [N] - bootstrap value for last step

    Returns:
        advantages: [T] or [T, N]
        returns: [T] or [T, N]
    """
    # Ensure 2D: [T, N]
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)
        values = values.unsqueeze(1)
        dones = dones.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False

    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Bootstrap
    if next_value is None:
        next_value = torch.zeros(N, device=rewards.device)

    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
            next_done = torch.zeros(N, device=rewards.device, dtype=torch.bool)
        else:
            next_val = values[t + 1]
            next_done = dones[t + 1]

        delta = rewards[t] + gamma * next_val * (1 - next_done.float()) - values[t]
        gae = delta + gamma * gae_lambda * (1 - next_done.float()) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    if squeeze:
        advantages = advantages.squeeze(1)
        returns = returns.squeeze(1)

    return advantages, returns


def compute_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    next_value: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute discounted returns (no GAE)."""
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)
        values = values.unsqueeze(1)
        dones = dones.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False

    T, N = rewards.shape
    returns = torch.zeros_like(rewards)

    if next_value is None:
        next_value = torch.zeros(N, device=rewards.device)

    ret = next_value
    for t in reversed(range(T)):
        ret = rewards[t] + gamma * ret * (1 - dones[t].float())
        returns[t] = ret

    if squeeze:
        returns = returns.squeeze(1)

    return returns


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute explained variance: 1 - Var(y_true - y_pred) / Var(y_true)."""
    assert y_pred.shape == y_true.shape

    var_y = torch.var(y_true)
    if var_y == 0:
        return 0.0

    return 1.0 - torch.var(y_true - y_pred) / var_y


def compute_kl_divergence(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence between old and new policies."""
    # KL = E[old_log_prob - new_log_prob]
    return (old_log_probs - new_log_probs).mean()


def compute_clip_fraction(
    ratios: torch.Tensor,
    clip_range: float
) -> torch.Tensor:
    """Compute fraction of clipped samples."""
    clipped = (ratios < 1 - clip_range) | (ratios > 1 + clip_range)
    return clipped.float().mean()


def compute_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy from log probabilities."""
    return -(log_probs * log_probs.exp()).sum(dim=-1).mean()


def gaussian_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action: torch.Tensor
) -> torch.Tensor:
    """Log probability of Gaussian distribution."""
    std = log_std.exp()
    var = std ** 2
    log_prob = -0.5 * ((action - mean) ** 2 / var + 2 * log_std + np.log(2 * np.pi))
    return log_prob.sum(dim=-1)


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """Entropy of Gaussian distribution."""
    return (log_std + 0.5 * np.log(2 * np.pi * np.e)).sum(dim=-1).mean()


def evaluate_policy(
    policy,
    env,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback=None
) -> Dict[str, float]:
    """Evaluate policy for n episodes."""
    episode_rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        length = 0

        while not done:
            with torch.no_grad():
                if deterministic:
                    action = policy.predict(obs, deterministic=True)
                else:
                    action = policy.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            length += 1

            if render:
                env.render()

            if callback:
                callback(obs, action, reward, info)

        episode_rewards.append(total_reward)
        episode_lengths.append(length)

        if info.get("success", False):
            successes += 1

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": successes / n_episodes,
    }


def safe_mean(arr: List[float]) -> float:
    """Safe mean that returns 0 for empty list."""
    return np.mean(arr) if arr else 0.0


def safe_std(arr: List[float]) -> float:
    """Safe std that returns 0 for empty list."""
    return np.std(arr) if arr else 0.0