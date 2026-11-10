# Critic Network (Value Function)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from core.neural.encoders import create_encoder, initialize_weights, get_activation


@dataclass
class CriticOutput:
    """Output from critic forward pass."""
    value: torch.Tensor
    features: Optional[torch.Tensor] = None
    aux_values: Optional[Dict[str, torch.Tensor]] = None


class MLPValueHead(nn.Module):
    """MLP Value Head."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [256, 256],
        activation: str = "tanh",
        layer_norm: bool = False,
        output_size: int = 1,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(get_activation(activation))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.net = nn.Sequential(*layers)
        self.apply(lambda m: initialize_weights(m, init_type, gain))

        # Output layer initialization
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    """Single Critic Network."""

    def __init__(
        self,
        obs_dim: int,
        encoder_config: Dict[str, Any],
        hidden_sizes: List[int] = [256, 256],
        activation: str = "tanh",
        layer_norm: bool = False,
        clip_value: Optional[float] = None,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.clip_value = clip_value

        # Encoder
        self.encoder = create_encoder(encoder_config, input_shape=(obs_dim,))
        encoder_output_size = self.encoder.output_size

        # Value head
        self.value_head = MLPValueHead(
            input_size=encoder_output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            layer_norm=layer_norm,
            output_size=1,
            init_type=init_type,
            gain=gain
        )

        self.apply(lambda m: initialize_weights(m, init_type, gain))

    def forward(self, obs: torch.Tensor) -> CriticOutput:
        features = self.encoder(obs)
        value = self.value_head(features).squeeze(-1)

        if self.clip_value is not None:
            value = torch.clamp(value, -self.clip_value, self.clip_value)

        return CriticOutput(value=value, features=features)


class TwinCritic(nn.Module):
    """Twin Critics (for TD3/SAC-style algorithms)."""

    def __init__(
        self,
        obs_dim: int,
        encoder_config: Dict[str, Any],
        hidden_sizes: List[int] = [256, 256],
        activation: str = "tanh",
        layer_norm: bool = False,
        clip_value: Optional[float] = None,
        shared_encoder: bool = False,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.clip_value = clip_value
        self.shared_encoder = shared_encoder

        if shared_encoder:
            # Shared encoder
            self.encoder = create_encoder(encoder_config, input_shape=(obs_dim,))
            encoder_output_size = self.encoder.output_size

            self.value_head1 = MLPValueHead(
                input_size=encoder_output_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
                layer_norm=layer_norm,
                output_size=1,
                init_type=init_type,
                gain=gain
            )
            self.value_head2 = MLPValueHead(
                input_size=encoder_output_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
                layer_norm=layer_norm,
                output_size=1,
                init_type=init_type,
                gain=gain
            )
        else:
            # Separate encoders
            self.critic1 = Critic(
                obs_dim=obs_dim,
                encoder_config=encoder_config,
                hidden_sizes=hidden_sizes,
                activation=activation,
                layer_norm=layer_norm,
                clip_value=clip_value,
                init_type=init_type,
                gain=gain
            )
            self.critic2 = Critic(
                obs_dim=obs_dim,
                encoder_config=encoder_config,
                hidden_sizes=hidden_sizes,
                activation=activation,
                layer_norm=layer_norm,
                clip_value=clip_value,
                init_type=init_type,
                gain=gain
            )

        self.apply(lambda m: initialize_weights(m, init_type, gain))

    def forward(self, obs: torch.Tensor) -> Tuple[CriticOutput, CriticOutput]:
        if self.shared_encoder:
            features = self.encoder(obs)
            value1 = self.value_head1(features).squeeze(-1)
            value2 = self.value_head2(features).squeeze(-1)

            if self.clip_value is not None:
                value1 = torch.clamp(value1, -self.clip_value, self.clip_value)
                value2 = torch.clamp(value2, -self.clip_value, self.clip_value)

            return (
                CriticOutput(value=value1, features=features),
                CriticOutput(value=value2, features=features)
            )
        else:
            out1 = self.critic1(obs)
            out2 = self.critic2(obs)
            return out1, out2

    def forward_single(self, obs: torch.Tensor, critic_idx: int = 0) -> CriticOutput:
        """Forward pass through single critic (for target networks)."""
        if self.shared_encoder:
            features = self.encoder(obs)
            if critic_idx == 0:
                value = self.value_head1(features).squeeze(-1)
            else:
                value = self.value_head2(features).squeeze(-1)

            if self.clip_value is not None:
                value = torch.clamp(value, -self.clip_value, self.clip_value)

            return CriticOutput(value=value, features=features)
        else:
            return self.critic1(obs) if critic_idx == 0 else self.critic2(obs)

    def min_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get minimum of twin critics (for TD3/SAC)."""
        out1, out2 = self.forward(obs)
        return torch.min(out1.value, out2.value)


class EnsembleCritic(nn.Module):
    """Ensemble of Critics for uncertainty estimation."""

    def __init__(
        self,
        obs_dim: int,
        encoder_config: Dict[str, Any],
        hidden_sizes: List[int] = [256, 256],
        activation: str = "tanh",
        layer_norm: bool = False,
        clip_value: Optional[float] = None,
        ensemble_size: int = 5,
        shared_encoder: bool = False,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.clip_value = clip_value
        self.ensemble_size = ensemble_size
        self.shared_encoder = shared_encoder

        if shared_encoder:
            self.encoder = create_encoder(encoder_config, input_shape=(obs_dim,))
            encoder_output_size = self.encoder.output_size

            self.value_heads = nn.ModuleList([
                MLPValueHead(
                    input_size=encoder_output_size,
                    hidden_sizes=hidden_sizes,
                    activation=activation,
                    layer_norm=layer_norm,
                    output_size=1,
                    init_type=init_type,
                    gain=gain
                )
                for _ in range(ensemble_size)
            ])
        else:
            self.critics = nn.ModuleList([
                Critic(
                    obs_dim=obs_dim,
                    encoder_config=encoder_config,
                    hidden_sizes=hidden_sizes,
                    activation=activation,
                    layer_norm=layer_norm,
                    clip_value=clip_value,
                    init_type=init_type,
                    gain=gain
                )
                for _ in range(ensemble_size)
            ])

        self.apply(lambda m: initialize_weights(m, init_type, gain))

    def forward(self, obs: torch.Tensor) -> List[CriticOutput]:
        if self.shared_encoder:
            features = self.encoder(obs)
            values = [head(features).squeeze(-1) for head in self.value_heads]

            if self.clip_value is not None:
                values = [torch.clamp(v, -self.clip_value, self.clip_value) for v in values]

            return [CriticOutput(value=v, features=features) for v in values]
        else:
            return [critic(obs) for critic in self.critics]

    def mean_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get mean value across ensemble."""
        outputs = self.forward(obs)
        return torch.stack([o.value for o in outputs]).mean(dim=0)

    def std_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get std value across ensemble (uncertainty)."""
        outputs = self.forward(obs)
        return torch.stack([o.value for o in outputs]).std(dim=0)

    def min_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get minimum value (pessimistic)."""
        outputs = self.forward(obs)
        return torch.stack([o.value for o in outputs]).min(dim=0).values

    def max_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get maximum value (optimistic)."""
        outputs = self.forward(obs)
        return torch.stack([o.value for o in outputs]).max(dim=0).values


class StateActionCritic(nn.Module):
    """Q-function critic (state-action value)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        encoder_config: Dict[str, Any],
        hidden_sizes: List[int] = [256, 256],
        activation: str = "tanh",
        layer_norm: bool = False,
        clip_value: Optional[float] = None,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clip_value = clip_value

        # Encoder for observations
        self.encoder = create_encoder(encoder_config, input_shape=(obs_dim,))
        encoder_output_size = self.encoder.output_size

        # Combined network
        layers = []
        prev_size = encoder_output_size + action_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(get_activation(activation))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.q_net = nn.Sequential(*layers)
        self.apply(lambda m: initialize_weights(m, init_type, gain))

        nn.init.orthogonal_(self.q_net[-1].weight, gain=0.01)
        nn.init.constant_(self.q_net[-1].bias, 0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> CriticOutput:
        features = self.encoder(obs)
        combined = torch.cat([features, action], dim=-1)
        value = self.q_net(combined).squeeze(-1)

        if self.clip_value is not None:
            value = torch.clamp(value, -self.clip_value, self.clip_value)

        return CriticOutput(value=value, features=features)


def create_critic(config: Dict[str, Any], obs_dim: int, action_dim: int = 0) -> nn.Module:
    """Factory function to create critic from config."""
    critic_type = config.get("type", "mlp")

    common_args = {
        "obs_dim": obs_dim,
        "encoder_config": config.get("encoder", {"type": "mlp"}),
        "hidden_sizes": config.get("hidden_sizes", [256, 256]),
        "activation": config.get("activation", "tanh"),
        "layer_norm": config.get("layer_norm", False),
        "clip_value": config.get("clip_value"),
        "init_type": config.get("init_type", "orthogonal"),
        "gain": config.get("gain", 1.0)
    }

    if critic_type == "mlp":
        return Critic(**common_args)

    elif critic_type == "twin":
        return TwinCritic(
            **common_args,
            shared_encoder=config.get("shared_encoder", False)
        )

    elif critic_type == "ensemble":
        return EnsembleCritic(
            **common_args,
            ensemble_size=config.get("ensemble_size", 5),
            shared_encoder=config.get("shared_encoder", False)
        )

    elif critic_type == "state_action":
        return StateActionCritic(
            **common_args,
            action_dim=action_dim
        )

    else:
        raise ValueError(f"Unknown critic type: {critic_type}")