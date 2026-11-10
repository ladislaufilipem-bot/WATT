# PPO Update Step Logic
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from core.ppo.buffer import RolloutBuffer, RolloutBatch
from core.ppo.loss import PPOLosses, ppo_loss, compute_gae
from core.neural.actor import ActorOutput, GaussianActor, CategoricalActor
from core.neural.critic import Critic, CriticOutput
from core.utils.metrics import explained_variance


@dataclass
class UpdateStats:
    """Statistics from a PPO update."""
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    entropy: float
    learning_rate: float
    grad_norm: float


class PPOUpdater:
    """Handles PPO update steps."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        normalize_advantage: bool = True,
        device: str = "cpu",
        amp_enabled: bool = True,
        amp_dtype: str = "float16"
    ):
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.normalize_advantage = normalize_advantage
        self.device = device

        # AMP
        self.amp_enabled = amp_enabled and device.startswith("cuda")
        self.amp_dtype = getattr(torch, amp_dtype) if amp_dtype != "float16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        # Compile (PyTorch 2.0+)
        self._compiled = False

    def compile_models(self) -> None:
        """Compile models with torch.compile (PyTorch 2.0+)."""
        if hasattr(torch, 'compile') and not self._compiled:
            self.actor = torch.compile(self.actor, mode="reduce-overhead")
            self.critic = torch.compile(self.critic, mode="reduce-overhead")
            self._compiled = True

    def update(
        self,
        buffer: RolloutBuffer,
        num_epochs: int,
        batch_size: int,
        last_values: torch.Tensor,
        last_dones: torch.Tensor
    ) -> List[UpdateStats]:
        """
        Perform PPO update.

        Args:
            buffer: Rollout buffer with collected data
            num_epochs: Number of epochs to iterate over buffer
            batch_size: Minibatch size
            last_values: Value estimates for last observations [n_envs]
            last_dones: Done flags for last observations [n_envs]

        Returns:
            List of UpdateStats for each epoch
        """
        # Compute advantages
        buffer.compute_advantages(last_values, last_dones)

        all_stats = []

        for epoch in range(num_epochs):
            epoch_stats = []

            for batch in buffer.get_generator(batch_size, num_epochs=1):
                stats = self._update_step(batch)
                epoch_stats.append(stats)

                # Early stopping based on KL
                if self.target_kl is not None and stats.approx_kl > self.target_kl * 1.5:
                    print(f"Early stopping at epoch {epoch}: KL={stats.approx_kl:.4f} > {self.target_kl * 1.5:.4f}")
                    break

            # Average epoch stats
            if epoch_stats:
                avg_stats = UpdateStats(
                    policy_loss=sum(s.policy_loss for s in epoch_stats) / len(epoch_stats),
                    value_loss=sum(s.value_loss for s in epoch_stats) / len(epoch_stats),
                    entropy_loss=sum(s.entropy_loss for s in epoch_stats) / len(epoch_stats),
                    total_loss=sum(s.total_loss for s in epoch_stats) / len(epoch_stats),
                    approx_kl=sum(s.approx_kl for s in epoch_stats) / len(epoch_stats),
                    clip_fraction=sum(s.clip_fraction for s in epoch_stats) / len(epoch_stats),
                    explained_variance=sum(s.explained_variance for s in epoch_stats) / len(epoch_stats),
                    entropy=sum(s.entropy for s in epoch_stats) / len(epoch_stats),
                    learning_rate=epoch_stats[-1].learning_rate,
                    grad_norm=sum(s.grad_norm for s in epoch_stats) / len(epoch_stats)
                )
                all_stats.append(avg_stats)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        return all_stats

    def _update_step(self, batch: RolloutBatch) -> UpdateStats:
        """Single gradient step."""
        self.optimizer.zero_grad()

        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            # Actor forward
            # Flatten observations for actor
            obs_flat = {}
            for key, tensor in batch.observations.items():
                obs_flat[key] = tensor

            actor_output = self.actor(obs_flat)

            # Critic forward
            critic_output = self.critic(obs_flat)
            new_values = critic_output.value

            # Compute loss
            losses = ppo_loss(
                actor_output=actor_output,
                batch=batch,
                clip_range=self.clip_range,
                clip_range_vf=self.clip_range_vf,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                old_values=batch.values
            )

            # Add value loss from critic
            value_loss, _ = compute_value_loss(
                new_values=new_values,
                old_values=batch.values,
                returns=batch.returns,
                clip_range_vf=self.clip_range_vf
            )

            # Replace value loss in losses
            losses.value_loss = value_loss
            losses.total_loss = losses.policy_loss + self.vf_coef * value_loss + losses.entropy_loss

        # Backward pass with gradient scaling
        self.scaler.scale(losses.total_loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm
        )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Get learning rate
        lr = self.optimizer.param_groups[0]['lr']

        return UpdateStats(
            policy_loss=losses.policy_loss.item(),
            value_loss=losses.value_loss.item(),
            entropy_loss=losses.entropy_loss.item(),
            total_loss=losses.total_loss.item(),
            approx_kl=losses.approx_kl.item(),
            clip_fraction=losses.clip_fraction.item(),
            explained_variance=losses.explained_variance.item(),
            entropy=losses.entropy.item(),
            learning_rate=lr,
            grad_norm=grad_norm.item()
        )

    def update_recurrent(
        self,
        buffer: RolloutBuffer,
        num_epochs: int,
        batch_size: int,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        sequence_length: int
    ) -> List[UpdateStats]:
        """
        Update for recurrent policies.

        Args:
            sequence_length: Length of sequences for truncated BPTT
        """
        # TODO: Implement recurrent update with sequence batching
        raise NotImplementedError("Recurrent update not yet implemented")


def create_optimizer(
    actor: nn.Module,
    critic: nn.Module,
    config: Dict[str, Any]
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Create optimizer and scheduler from config."""
    optimizer_type = config.get("optimizer", {}).get("type", "adam")
    lr = config.get("learning_rate", 3e-4)
    eps = config.get("optimizer", {}).get("eps", 1e-5)
    weight_decay = config.get("optimizer", {}).get("weight_decay", 0.0)
    betas = config.get("optimizer", {}).get("betas", [0.9, 0.999])

    # Combine parameters
    params = list(actor.parameters()) + list(critic.parameters())

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, eps=eps, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=lr, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Scheduler
    lr_schedule = config.get("lr_schedule", {})
    schedule_type = lr_schedule.get("type", "constant")

    scheduler = None
    if schedule_type == "linear":
        # Need total steps - will be set later
        pass
    elif schedule_type == "cosine":
        pass
    elif schedule_type == "exponential":
        gamma = lr_schedule.get("gamma", 0.999)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif schedule_type == "constant":
        pass

    return optimizer, scheduler


class PPOTrainer:
    """High-level PPO Trainer."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        buffer: RolloutBuffer,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        self.actor = actor
        self.critic = critic
        self.buffer = buffer
        self.config = config
        self.device = device

        # Create optimizer
        self.optimizer, self.scheduler = create_optimizer(actor, critic, config)

        # Create updater
        self.updater = PPOUpdater(
            actor=actor,
            critic=critic,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            clip_range=config.get("clip_range", 0.2),
            clip_range_vf=config.get("clip_range_vf"),
            ent_coef=config.get("ent_coef", 0.01),
            vf_coef=config.get("vf_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            target_kl=config.get("target_kl"),
            normalize_advantage=config.get("normalize_advantage", True),
            device=device,
            amp_enabled=config.get("amp", {}).get("enabled", True),
            amp_dtype=config.get("amp", {}).get("dtype", "float16")
        )

        # Compile if requested
        if config.get("compile", False):
            self.updater.compile_models()

    def train_step(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor
    ) -> List[UpdateStats]:
        """Perform one training update."""
        return self.updater.update(
            buffer=self.buffer,
            num_epochs=self.config.get("n_epochs", 10),
            batch_size=self.config.get("batch_size", 64),
            last_values=last_values,
            last_dones=last_dones
        )

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "updater_scaler": self.updater.scaler.state_dict()
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        if "updater_scaler" in state:
            self.updater.scaler.load_state_dict(state["updater_scaler"])