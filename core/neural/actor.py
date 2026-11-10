# Actor Network (Policy)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import numpy as np

from core.neural.encoders import create_encoder, initialize_weights, get_activation


@dataclass
class ActorOutput:
    """Output from actor forward pass."""
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    mean: Optional[torch.Tensor] = None
    log_std: Optional[torch.Tensor] = None
    dist_params: Optional[Dict[str, torch.Tensor]] = None


class GaussianActor(nn.Module):
    """Gaussian policy for continuous action spaces."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        encoder_config: Dict[str, Any],
        hidden_sizes: List[int] = [256, 256],
        activation: str = "tanh",
        layer_norm: bool = False,
        log_std_init: float = -0.5,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        state_dependent_std: bool = True,
        squash: bool = False,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.state_dependent_std = state_dependent_std
        self.squash = squash
        self.action_scale = action_scale
        self.action_bias = action_bias

        # Encoder
        self.encoder = create_encoder(encoder_config, input_shape=(obs_dim,))
        encoder_output_size = self.encoder.output_size

        # Policy network
        layers = []
        prev_size = encoder_output_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(get_activation(activation))
            prev_size = hidden_size

        self.policy_net = nn.Sequential(*layers)

        # Mean head
        self.mean_head = nn.Linear(prev_size, action_dim)

        # Log std head
        if state_dependent_std:
            self.log_std_head = nn.Linear(prev_size, action_dim)
        else:
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

        # Action scaling
        self.register_buffer("action_scale_tensor", torch.tensor(action_scale))
        self.register_buffer("action_bias_tensor", torch.tensor(action_bias))

        self.apply(lambda m: initialize_weights(m, init_type, gain))

        # Special initialization for output layers
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0)

        if state_dependent_std:
            nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
            nn.init.constant_(self.log_std_head.bias, log_std_init)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action: Optional[torch.Tensor] = None
    ) -> ActorOutput:
        """
        Args:
            obs: [B, obs_dim]
            deterministic: If True, return mean action
            action: If provided, compute log_prob for this action

        Returns:
            ActorOutput with action, log_prob, entropy
        """
        # Encode observations
        features = self.encoder(obs)
        features = self.policy_net(features)

        # Mean
        mean = self.mean_head(features)

        # Log std
        if self.state_dependent_std:
            log_std = self.log_std_head(features)
        else:
            log_std = self.log_std.expand_as(mean)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        # Distribution
        dist = torch.distributions.Normal(mean, std)

        # Sample or use provided action
        if action is None:
            if deterministic:
                action = mean
            else:
                action = dist.rsample()  # Reparameterized sample

        # Compute log prob
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Entropy
        entropy = dist.entropy().sum(dim=-1).mean()

        # Squash correction (for SAC-style)
        if self.squash:
            # tanh squashing
            action = torch.tanh(action)
            # Correction: log(1 - tanh^2)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        # Scale and bias
        action = action * self.action_scale_tensor + self.action_bias_tensor

        return ActorOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            mean=mean,
            log_std=log_std,
            dist_params={"mean": mean, "std": std}
        )

    def get_distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Get distribution for obs."""
        features = self.encoder(obs)
        features = self.policy_net(features)
        mean = self.mean_head(features)

        if self.state_dependent_std:
            log_std = self.log_std_head(features)
        else:
            log_std = self.log_std.expand_as(mean)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return torch.distributions.Normal(mean, log_std.exp())

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict action (for deployment)."""
        with torch.no_grad():
            output = self.forward(obs, deterministic=deterministic)
        return output.action


class SquashedGaussianActor(GaussianActor):
    """Squashed Gaussian policy (SAC-style) with tanh squashing."""

    def __init__(self, *args, **kwargs):
        kwargs["squash"] = True
        super().__init__(*args, **kwargs)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action: Optional[torch.Tensor] = None
    ) -> ActorOutput:
        # Get pre-squash action
        features = self.encoder(obs)
        features = self.policy_net(features)
        mean = self.mean_head(features)

        if self.state_dependent_std:
            log_std = self.log_std_head(features)
        else:
            log_std = self.log_std.expand_as(mean)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)

        if action is None:
            if deterministic:
                pre_squash = mean
            else:
                pre_squash = dist.rsample()
        else:
            # Inverse squash: atanh(action)
            pre_squash = torch.atanh(torch.clamp(action, -1 + 1e-6, 1 - 1e-6))

        # Log prob with correction
        log_prob = dist.log_prob(pre_squash).sum(dim=-1)
        log_prob -= torch.log(1 - torch.tanh(pre_squash).pow(2) + 1e-6).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1).mean()

        # Squash
        squashed = torch.tanh(pre_squash)
        action_out = squashed * self.action_scale_tensor + self.action_bias_tensor

        return ActorOutput(
            action=action_out,
            log_prob=log_prob,
            entropy=entropy,
            mean=torch.tanh(mean) * self.action_scale_tensor + self.action_bias_tensor,
            log_std=log_std,
            dist_params={"mean": mean, "std": std, "squashed": squashed}
        )


class CategoricalActor(nn.Module):
    """Categorical policy for discrete action spaces."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        encoder_config: Dict[str, Any],
        hidden_sizes: List[int] = [256, 256],
        activation: str = "tanh",
        layer_norm: bool = False,
        temperature: float = 1.0,
        gumbel_softmax: bool = False,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.gumbel_softmax = gumbel_softmax

        # Encoder
        self.encoder = create_encoder(encoder_config, input_shape=(obs_dim,))
        encoder_output_size = self.encoder.output_size

        # Policy network
        layers = []
        prev_size = encoder_output_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(get_activation(activation))
            prev_size = hidden_size

        self.policy_net = nn.Sequential(*layers)

        # Logits head
        self.logits_head = nn.Linear(prev_size, action_dim)

        self.apply(lambda m: initialize_weights(m, init_type, gain))

        nn.init.orthogonal_(self.logits_head.weight, gain=0.01)
        nn.init.constant_(self.logits_head.bias, 0)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action: Optional[torch.Tensor] = None
    ) -> ActorOutput:
        features = self.encoder(obs)
        features = self.policy_net(features)
        logits = self.logits_head(features) / self.temperature

        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            elif self.gumbel_softmax:
                action = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
                action = action.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()

        return ActorOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            mean=logits,
            dist_params={"logits": logits}
        )

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            output = self.forward(obs, deterministic=deterministic)
        return output.action


class MixtureActor(nn.Module):
    """Mixture of Gaussians policy."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        encoder_config: Dict[str, Any],
        hidden_sizes: List[int] = [256, 256],
        activation: str = "tanh",
        layer_norm: bool = False,
        num_components: int = 5,
        component_type: str = "gaussian",
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_components = num_components

        # Encoder
        self.encoder = create_encoder(encoder_config, input_shape=(obs_dim,))
        encoder_output_size = self.encoder.output_size

        # Policy network
        layers = []
        prev_size = encoder_output_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(get_activation(activation))
            prev_size = hidden_size

        self.policy_net = nn.Sequential(*layers)

        # Mixture heads
        self.mixing_head = nn.Linear(prev_size, num_components)
        self.mean_heads = nn.ModuleList([
            nn.Linear(prev_size, action_dim) for _ in range(num_components)
        ])
        self.log_std_heads = nn.ModuleList([
            nn.Linear(prev_size, action_dim) for _ in range(num_components)
        ])

        self.apply(lambda m: initialize_weights(m, init_type, gain))

        for head in self.mean_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.constant_(head.bias, 0)
        for head in self.log_std_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.constant_(head.bias, -0.5)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action: Optional[torch.Tensor] = None
    ) -> ActorOutput:
        features = self.encoder(obs)
        features = self.policy_net(features)

        # Mixing weights
        mixing_logits = self.mixing_head(features)
        mixing_weights = F.softmax(mixing_logits, dim=-1)

        # Component parameters
        means = torch.stack([head(features) for head in self.mean_heads], dim=1)
        log_stds = torch.stack([head(features) for head in self.log_std_heads], dim=1)
        log_stds = torch.clamp(log_stds, -20.0, 2.0)
        stds = log_stds.exp()

        # Mixture distribution
        components = [
            torch.distributions.Normal(means[:, i], stds[:, i])
            for i in range(self.num_components)
        ]

        mixture = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(mixing_weights),
            component_distribution=torch.distributions.Independent(
                torch.distributions.Normal(means, stds), 1
            )
        )

        if action is None:
            if deterministic:
                # Use highest weight component mean
                best_component = mixing_weights.argmax(dim=-1)
                action = means[torch.arange(means.size(0)), best_component]
            else:
                action = mixture.sample()

        log_prob = mixture.log_prob(action)
        entropy = mixture.entropy().mean()

        return ActorOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            mean=means,
            log_std=log_stds,
            dist_params={
                "mixing_weights": mixing_weights,
                "means": means,
                "stds": stds
            }
        )

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            output = self.forward(obs, deterministic=deterministic)
        return output.action


def create_actor(config: Dict[str, Any], obs_dim: int, action_dim: int) -> nn.Module:
    """Factory function to create actor from config."""
    actor_type = config.get("type", "gaussian")

    common_args = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "encoder_config": config.get("encoder", {"type": "mlp"}),
        "hidden_sizes": config.get("hidden_sizes", [256, 256]),
        "activation": config.get("activation", "tanh"),
        "layer_norm": config.get("layer_norm", False),
        "init_type": config.get("init_type", "orthogonal"),
        "gain": config.get("gain", 1.0)
    }

    if actor_type == "gaussian":
        return GaussianActor(
            **common_args,
            log_std_init=config.get("log_std_init", -0.5),
            log_std_min=config.get("log_std_min", -20.0),
            log_std_max=config.get("log_std_max", 2.0),
            state_dependent_std=config.get("state_dependent_std", True),
            squash=config.get("squash", False),
            action_scale=config.get("action_scale", 1.0),
            action_bias=config.get("action_bias", 0.0)
        )

    elif actor_type == "squashed_gaussian":
        return SquashedGaussianActor(
            **common_args,
            log_std_init=config.get("log_std_init", -0.5),
            log_std_min=config.get("log_std_min", -20.0),
            log_std_max=config.get("log_std_max", 2.0),
            state_dependent_std=config.get("state_dependent_std", True),
            action_scale=config.get("action_scale", 1.0),
            action_bias=config.get("action_bias", 0.0)
        )

    elif actor_type == "categorical":
        return CategoricalActor(
            **common_args,
            temperature=config.get("temperature", 1.0),
            gumbel_softmax=config.get("gumbel_softmax", False)
        )

    elif actor_type == "mixture":
        return MixtureActor(
            **common_args,
            num_components=config.get("num_components", 5),
            component_type=config.get("component_type", "gaussian")
        )

    else:
        raise ValueError(f"Unknown actor type: {actor_type}")