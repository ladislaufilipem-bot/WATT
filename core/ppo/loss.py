# PPO Loss Functions
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from core.ppo.buffer import RolloutBatch
from core.neural.actor import ActorOutput


@dataclass
class PPOLosses:
    """Container for PPO loss components."""
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    total_loss: torch.Tensor
    # Metrics
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    explained_variance: torch.Tensor
    entropy: torch.Tensor


def compute_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    clip_range_vf: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PPO policy loss with clipping.

    Returns:
        policy_loss, clip_fraction, approx_kl
    """
    # Ratio
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Clipped surrogate objective
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss_1 = ratio * advantages
    policy_loss_2 = clipped_ratio * advantages
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    # Clip fraction
    clip_fraction = ((ratio < 1 - clip_range) | (ratio > 1 + clip_range)).float().mean()

    # Approximate KL divergence
    approx_kl = (old_log_probs - new_log_probs).mean()

    return policy_loss, clip_fraction, approx_kl


def compute_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_range_vf: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute value function loss.

    Returns:
        value_loss, explained_variance
    """
    if clip_range_vf is not None:
        # Clipped value loss
        value_clipped = old_values + torch.clamp(
            new_values - old_values,
            -clip_range_vf,
            clip_range_vf
        )
        value_loss_1 = (new_values - returns) ** 2
        value_loss_2 = (value_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
    else:
        value_loss = 0.5 * F.mse_loss(new_values, returns)

    # Explained variance
    var_returns = torch.var(returns)
    if var_returns > 0:
        explained_variance = 1.0 - torch.var(returns - new_values) / var_returns
    else:
        explained_variance = torch.tensor(0.0, device=returns.device)

    return value_loss, explained_variance


def compute_entropy_loss(
    entropy: torch.Tensor,
    ent_coef: float
) -> torch.Tensor:
    """Compute entropy bonus loss."""
    return -ent_coef * entropy


def ppo_loss(
    actor_output: ActorOutput,
    batch: RolloutBatch,
    clip_range: float,
    clip_range_vf: Optional[float],
    ent_coef: float,
    vf_coef: float,
    old_values: Optional[torch.Tensor] = None
) -> PPOLosses:
    """
    Compute full PPO loss.

    Args:
        actor_output: Output from actor forward pass (with new log_probs)
        batch: Rollout batch with old values
        clip_range: PPO clip range
        clip_range_vf: Value function clip range (None = no clipping)
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        old_values: Old value estimates (for clipped value loss)

    Returns:
        PPOLosses with all components
    """
    # Policy loss
    policy_loss, clip_fraction, approx_kl = compute_policy_loss(
        new_log_probs=actor_output.log_prob,
        old_log_probs=batch.log_probs,
        advantages=batch.advantages,
        clip_range=clip_range
    )

    # Value loss
    # We need the critic's new values - these come from a separate forward pass
    # For now, assume they're computed externally and passed via batch
    # Or compute them here if we have the critic
    if hasattr(batch, 'new_values') and batch.new_values is not None:
        new_values = batch.new_values
    else:
        # Fallback: use batch values (not ideal)
        new_values = batch.values

    old_values_for_vf = old_values if old_values is not None else batch.values

    value_loss, explained_variance = compute_value_loss(
        new_values=new_values,
        old_values=old_values_for_vf,
        returns=batch.returns,
        clip_range_vf=clip_range_vf
    )

    # Entropy loss
    entropy_loss = compute_entropy_loss(actor_output.entropy, ent_coef)

    # Total loss
    total_loss = policy_loss + vf_coef * value_loss + entropy_loss

    return PPOLosses(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        total_loss=total_loss,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        explained_variance=explained_variance,
        entropy=actor_output.entropy
    )


class PPOLossModule(nn.Module):
    """PPO Loss as a module for easy integration."""

    def __init__(
        self,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        normalize_advantage: bool = True
    ):
        super().__init__()
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.normalize_advantage = normalize_advantage

    def forward(
        self,
        actor_output: ActorOutput,
        batch: RolloutBatch,
        new_values: torch.Tensor,
        old_values: Optional[torch.Tensor] = None
    ) -> PPOLosses:
        # Normalize advantages if needed
        if self.normalize_advantage and not hasattr(batch, '_advantages_normalized'):
            batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            batch._advantages_normalized = True

        return ppo_loss(
            actor_output=actor_output,
            batch=batch,
            clip_range=self.clip_range,
            clip_range_vf=self.clip_range_vf,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            old_values=old_values
        )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    next_value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: [T, N] rewards
        values: [T, N] value estimates
        dones: [T, N] done flags
        gamma: Discount factor
        gae_lambda: GAE lambda
        next_value: [N] bootstrap value

    Returns:
        advantages: [T, N]
        returns: [T, N]
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

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

    return advantages, returns


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute explained variance: 1 - Var(y - y_hat) / Var(y)

    Args:
        pred: Predicted values
        target: Target values

    Returns:
        Explained variance (1.0 = perfect, 0.0 = no better than mean, negative = worse)
    """
    var_y = torch.var(target)
    if var_y == 0:
        return torch.tensor(0.0, device=pred.device)
    return 1.0 - torch.var(target - pred) / var_y


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0
) -> torch.Tensor:
    """Huber loss for value function."""
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < delta,
        0.5 * diff ** 2,
        delta * (diff - 0.5 * delta)
    )
    return loss.mean()


class ClippedValueLoss(nn.Module):
    """Clipped value loss module."""

    def __init__(self, clip_range: float = 10.0):
        super().__init__()
        self.clip_range = clip_range

    def forward(
        self,
        new_values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        value_clipped = old_values + torch.clamp(
            new_values - old_values,
            -self.clip_range,
            self.clip_range
        )
        loss_1 = (new_values - returns) ** 2
        loss_2 = (value_clipped - returns) ** 2
        return 0.5 * torch.max(loss_1, loss_2).mean()


class AdaptiveKLController:
    """Adaptive KL penalty controller (from PPO paper)."""

    def __init__(
        self,
        target_kl: float = 0.01,
        initial_kl_coef: float = 0.2,
        kl_penalty: bool = False
    ):
        self.target_kl = target_kl
        self.kl_coef = initial_kl_coef
        self.kl_penalty = kl_penalty

    def update(self, kl: float) -> None:
        """Update KL coefficient based on measured KL."""
        if self.kl_penalty:
            if kl > self.target_kl * 1.5:
                self.kl_coef *= 2.0
            elif kl < self.target_kl / 1.5:
                self.kl_coef *= 0.5
            # Clamp
            self.kl_coef = max(1e-6, min(10.0, self.kl_coef))

    def get_kl_loss(self, kl: torch.Tensor) -> torch.Tensor:
        """Get KL penalty loss."""
        if self.kl_penalty:
            return self.kl_coef * kl
        return torch.tensor(0.0, device=kl.device)