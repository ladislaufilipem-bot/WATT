# Rollout Buffer for PPO
from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class RolloutBatch:
    """Single batch from rollout buffer."""
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    # Optional
    next_observations: Optional[Dict[str, torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


class RolloutBuffer:
    """
    Rollout buffer for PPO.

    Stores trajectories and computes GAE advantages.
    Supports both single and vectorized environments.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_space: Dict[str, Tuple[int, ...]],
        action_space: Tuple[int, ...],
        n_envs: int = 1,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True
    ):
        self.buffer_size = buffer_size
        self.obs_space = obs_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantage = normalize_advantage

        # Storage
        self.observations: Dict[str, torch.Tensor] = {}
        self.next_observations: Dict[str, torch.Tensor] = {}
        self.actions = torch.zeros((buffer_size, n_envs) + action_space, dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, n_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, n_envs), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((buffer_size, n_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, n_envs), dtype=torch.bool, device=device)
        self.advantages = torch.zeros((buffer_size, n_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size, n_envs), dtype=torch.float32, device=device)

        # For recurrent policies
        self.hidden_states: List[Tuple[torch.Tensor, ...]] = []

        # Episode tracking
        self.episode_starts = torch.zeros((buffer_size, n_envs), dtype=torch.bool, device=device)
        self.episode_rewards = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.episode_lengths = torch.zeros(n_envs, dtype=torch.int32, device=device)

        # Initialize observation buffers
        for key, shape in obs_space.items():
            self.observations[key] = torch.zeros((buffer_size, n_envs) + shape, dtype=torch.float32, device=device)
            self.next_observations[key] = torch.zeros((buffer_size, n_envs) + shape, dtype=torch.float32, device=device)

        self.pos = 0
        self.full = False
        self.generator_ready = False

    def reset(self) -> None:
        """Reset buffer."""
        self.pos = 0
        self.full = False
        self.generator_ready = False
        self.episode_rewards.zero_()
        self.episode_lengths.zero_()
        self.hidden_states.clear()

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
        next_obs: Optional[Dict[str, torch.Tensor]] = None,
        hidden_state: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> None:
        """
        Add transition to buffer.

        Args:
            obs: Dict of observation tensors [n_envs, ...]
            action: Action tensor [n_envs, action_dim]
            reward: Reward tensor [n_envs]
            value: Value estimate [n_envs]
            log_prob: Log probability [n_envs]
            done: Done flags [n_envs]
            next_obs: Next observations (for terminal states)
            hidden_state: RNN hidden state
        """
        # Store observations
        for key, tensor in obs.items():
            self.observations[key][self.pos].copy_(tensor)

        # Store next observations
        if next_obs is not None:
            for key, tensor in next_obs.items():
                self.next_observations[key][self.pos].copy_(tensor)

        # Store transition
        self.actions[self.pos].copy_(action)
        self.rewards[self.pos].copy_(reward)
        self.values[self.pos].copy_(value)
        self.log_probs[self.pos].copy_(log_prob)
        self.dones[self.pos].copy_(done)

        # Episode tracking
        self.episode_starts[self.pos] = (self.pos == 0) | self.dones[self.pos - 1] if self.pos > 0 else True
        self.episode_rewards += reward.cpu()
        self.episode_lengths += 1

        # Store hidden state for recurrent policies
        if hidden_state is not None:
            self.hidden_states.append(hidden_state)

        # Reset episode stats for done envs
        done_mask = done.cpu().bool()
        if done_mask.any():
            self.episode_rewards[done_mask] = 0
            self.episode_lengths[done_mask] = 0

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor
    ) -> None:
        """
        Compute GAE advantages and returns.

        Args:
            last_values: Value estimates for next state [n_envs]
            last_dones: Done flags for next state [n_envs]
        """
        # Handle wrap-around
        if self.full:
            indices = torch.arange(self.buffer_size, device=self.device)
        else:
            indices = torch.arange(self.pos, device=self.device)

        # Get last value and done for each env
        next_value = last_values
        next_done = last_dones

        # Compute in reverse
        gae = 0.0
        for t in reversed(indices.tolist()):
            if t == self.buffer_size - 1 or (not self.full and t == self.pos - 1):
                # Last step in buffer
                mask = 1.0 - next_done.float()
                delta = self.rewards[t] + self.gamma * next_value * mask - self.values[t]
            else:
                mask = 1.0 - self.dones[t + 1].float()
                delta = self.rewards[t] + self.gamma * self.values[t + 1] * mask - self.values[t]

            gae = delta + self.gamma * self.gae_lambda * mask * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

            next_value = self.values[t]
            next_done = self.dones[t]

        self.generator_ready = True

    def get_batch(self, indices: torch.Tensor) -> RolloutBatch:
        """Get batch by indices."""
        batch_obs = {k: v[indices] for k, v in self.observations.items()}
        batch_next_obs = {k: v[indices] for k, v in self.next_observations.items()}

        return RolloutBatch(
            observations=batch_obs,
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            values=self.values[indices],
            log_probs=self.log_probs[indices],
            dones=self.dones[indices],
            advantages=self.advantages[indices],
            returns=self.returns[indices],
            next_observations=batch_next_obs,
            hidden_states=self.hidden_states[indices[0]] if self.hidden_states and indices.numel() > 0 else None
        )

    def get_generator(
        self,
        batch_size: int,
        num_epochs: int = 1,
        shuffle: bool = True
    ):
        """Generate minibatches for PPO epochs."""
        if not self.generator_ready:
            raise RuntimeError("Call compute_advantages() before getting generator")

        total_samples = self.buffer_size if self.full else self.pos
        indices = torch.arange(total_samples * self.n_envs, device=self.device)

        # Reshape to [T, N] -> [T*N]
        # Flatten: [buffer_size, n_envs, ...] -> [buffer_size * n_envs, ...]
        for key in self.observations:
            self.observations[key] = self.observations[key].view(-1, *self.observations[key].shape[2:])
        for key in self.next_observations:
            self.next_observations[key] = self.next_observations[key].view(-1, *self.next_observations[key].shape[2:])

        self.actions = self.actions.view(-1, *self.actions.shape[2:])
        self.rewards = self.rewards.view(-1)
        self.values = self.values.view(-1)
        self.log_probs = self.log_probs.view(-1)
        self.dones = self.dones.view(-1)
        self.advantages = self.advantages.view(-1)
        self.returns = self.returns.view(-1)

        # Normalize advantages
        if self.normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        for _ in range(num_epochs):
            if shuffle:
                perm = torch.randperm(total_samples * self.n_envs, device=self.device)
            else:
                perm = indices

            for start in range(0, len(perm), batch_size):
                end = min(start + batch_size, len(perm))
                batch_indices = perm[start:end]
                yield self.get_batch(batch_indices)

        # Restore shapes
        if self.full:
            t_dim = self.buffer_size
        else:
            t_dim = self.pos

        for key in self.observations:
            self.observations[key] = self.observations[key].view(t_dim, self.n_envs, *self.observations[key].shape[1:])
        for key in self.next_observations:
            self.next_observations[key] = self.next_observations[key].view(t_dim, self.n_envs, *self.next_observations[key].shape[1:])

        self.actions = self.actions.view(t_dim, self.n_envs, *self.actions.shape[1:])
        self.rewards = self.rewards.view(t_dim, self.n_envs)
        self.values = self.values.view(t_dim, self.n_envs)
        self.log_probs = self.log_probs.view(t_dim, self.n_envs)
        self.dones = self.dones.view(t_dim, self.n_envs)
        self.advantages = self.advantages.view(t_dim, self.n_envs)
        self.returns = self.returns.view(t_dim, self.n_envs)

    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics for completed episodes."""
        # Find episode boundaries
        done_indices = self.dones.nonzero(as_tuple=True)[0]

        if len(done_indices) == 0:
            return {}

        episode_rewards = []
        episode_lengths = []

        start = 0
        for idx in done_indices:
            if self.full:
                episode_rewards.append(self.rewards[start:idx+1].sum().item())
                episode_lengths.append((idx - start + 1).item())
            else:
                if idx < self.pos:
                    episode_rewards.append(self.rewards[start:idx+1].sum().item())
                    episode_lengths.append((idx - start + 1).item())
            start = idx + 1

        if episode_rewards:
            return {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "mean_length": np.mean(episode_lengths),
                "std_length": np.std(episode_lengths),
                "num_episodes": len(episode_rewards)
            }
        return {}

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def capacity(self) -> int:
        return self.buffer_size * self.n_envs


class DictRolloutBuffer(RolloutBuffer):
    """Rollout buffer with dictionary observations (alias for clarity)."""
    pass


def create_buffer(
    config: Dict[str, Any],
    obs_space: Dict[str, Tuple[int, ...]],
    action_space: Tuple[int, ...],
    n_envs: int = 1,
    device: str = "cpu"
) -> RolloutBuffer:
    """Factory function to create rollout buffer."""
    return RolloutBuffer(
        buffer_size=config.get("n_steps", 2048),
        obs_space=obs_space,
        action_space=action_space,
        n_envs=n_envs,
        device=device,
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        normalize_advantage=config.get("normalize_advantage", True)
    )