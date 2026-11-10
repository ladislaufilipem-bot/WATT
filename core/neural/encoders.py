# Neural Network Encoders
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import math


def get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }

    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")

    return activations[name]()


def initialize_weights(module: nn.Module, init_type: str = "orthogonal", gain: float = 1.0) -> None:
    """Initialize module weights."""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if init_type == "orthogonal":
            nn.init.orthogonal_(module.weight, gain=gain)
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(module.weight, gain=gain)
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        else:
            raise ValueError(f"Unknown init type: {init_type}")

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        for name, param in module.named_parameters():
            if 'weight' in name:
                if init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=gain)
                else:
                    nn.init.xavier_uniform_(param, gain=gain)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class MLPEncoder(nn.Module):
    """Multi-Layer Perceptron Encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int] = [128, 128],
        activation: str = "tanh",
        layer_norm: bool = False,
        output_size: Optional[int] = None,
        output_activation: Optional[str] = None,
        dropout: float = 0.0,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size or hidden_sizes[-1]

        layers = []
        prev_size = input_dim

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))

            layers.append(get_activation(activation))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_size = hidden_size

        # Output layer
        if output_size is not None and output_size != hidden_sizes[-1]:
            layers.append(nn.Linear(prev_size, output_size))
            prev_size = output_size

        if output_activation:
            layers.append(get_activation(output_activation))

        self.net = nn.Sequential(*layers)
        self.apply(lambda m: initialize_weights(m, init_type, gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNEncoder(nn.Module):
    """Convolutional Neural Network Encoder for images."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (C, H, W)
        channels: List[int] = [32, 64, 64],
        kernels: List[int] = [8, 4, 3],
        strides: List[int] = [4, 2, 1],
        paddings: Optional[List[int]] = None,
        activation: str = "relu",
        layer_norm: bool = False,
        output_size: int = 512,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.input_shape = input_shape
        C, H, W = input_shape

        if paddings is None:
            paddings = [0] * len(channels)

        layers = []
        in_channels = C

        for i, (out_channels, kernel, stride, padding) in enumerate(
            zip(channels, kernels, strides, paddings)
        ):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel, stride, padding))

            if layer_norm:
                layers.append(nn.GroupNorm(1, out_channels))  # LayerNorm equivalent

            layers.append(get_activation(activation))

            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.conv(dummy)
            self.conv_output_size = conv_out.numel()

        self.fc = nn.Linear(self.conv_output_size, output_size)
        self.output_activation = get_activation("relu")  # Usually ReLU after conv

        self.apply(lambda m: initialize_weights(m, init_type, gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return self.output_activation(x)


class RNNEncoder(nn.Module):
    """Recurrent Encoder (LSTM/GRU)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        rnn_class = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[rnn_type.lower()]

        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=batch_first
        )

        self.output_size = hidden_size * (2 if bidirectional else 1)

        self.apply(lambda m: initialize_weights(m, init_type, gain))

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            x: [B, T, input_size] if batch_first else [T, B, input_size]
            hidden: Optional initial hidden state

        Returns:
            output: [B, T, output_size] or [T, B, output_size]
            hidden: Final hidden state
        """
        output, hidden = self.rnn(x, hidden)
        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Initialize hidden state."""
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=device)

        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros_like(h0)
            return (h0, c0)
        return (h0,)


class TransformerEncoder(nn.Module):
    """Transformer Encoder for sequence data."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
        norm_first: bool = True,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.batch_first = batch_first

        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_size = d_model

        self.apply(lambda m: initialize_weights(m, init_type, gain))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_size] if batch_first else [T, B, input_size]
            mask: Attention mask [T, T] or [B, T, T]
            src_key_padding_mask: [B, T] padding mask
        """
        x = self.input_proj(x)
        x = self.transformer(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return x


class LidarEncoder(nn.Module):
    """1D Convolutional Encoder for LiDAR/range data."""

    def __init__(
        self,
        num_rays: int = 360,
        input_channels: int = 1,
        channels: List[int] = [32, 64, 128],
        kernels: List[int] = [5, 5, 3],
        strides: List[int] = [2, 2, 1],
        activation: str = "relu",
        pool: str = "max",  # max, avg, none
        output_size: int = 128,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.num_rays = num_rays

        layers = []
        in_channels = input_channels

        for i, (out_channels, kernel, stride) in enumerate(zip(channels, kernels, strides)):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel, stride))

            if pool != "none" and i < len(channels) - 1:
                if pool == "max":
                    layers.append(nn.MaxPool1d(2))
                elif pool == "avg":
                    layers.append(nn.AvgPool1d(2))

            layers.append(get_activation(activation))

            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, num_rays)
            conv_out = self.conv(dummy)
            self.conv_output_size = conv_out.numel()

        self.fc = nn.Linear(self.conv_output_size, output_size)
        self.output_activation = get_activation("relu")

        self.apply(lambda m: initialize_weights(m, init_type, gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_rays] or [B, 1, num_rays]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, num_rays]

        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return self.output_activation(x)


class MultiModalFusion(nn.Module):
    """Fusion module for multi-modal observations."""

    def __init__(
        self,
        modality_encoders: Dict[str, nn.Module],
        fusion_type: str = "concat",  # concat, attention, cross_attention, film
        attention_heads: int = 4,
        attention_dim: int = 256,
        output_size: int = 256,
        init_type: str = "orthogonal",
        gain: float = 1.0
    ):
        super().__init__()

        self.modality_encoders = nn.ModuleDict(modality_encoders)
        self.fusion_type = fusion_type
        self.modalities = list(modality_encoders.keys())

        # Compute total input size
        total_size = sum(enc.output_size for enc in modality_encoders.values())

        if fusion_type == "concat":
            self.fusion = nn.Linear(total_size, output_size)
            self.output_activation = get_activation("relu")

        elif fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=attention_dim,
                num_heads=attention_heads,
                batch_first=True
            )
            # Project each modality to attention_dim
            self.modality_projs = nn.ModuleDict({
                name: nn.Linear(enc.output_size, attention_dim)
                for name, enc in modality_encoders.items()
            })
            self.fusion = nn.Linear(attention_dim, output_size)
            self.output_activation = get_activation("relu")

        elif fusion_type == "film":
            # Feature-wise Linear Modulation
            # Use first modality to modulate others
            primary = self.modalities[0]
            primary_size = modality_encoders[primary].output_size
            other_size = sum(enc.output_size for name, enc in modality_encoders.items() if name != primary)

            self.gamma = nn.Linear(primary_size, other_size)
            self.beta = nn.Linear(primary_size, other_size)
            self.fusion = nn.Linear(total_size, output_size)
            self.output_activation = get_activation("relu")

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.apply(lambda m: initialize_weights(m, init_type, gain))

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            observations: Dict of modality_name -> tensor
        """
        # Encode each modality
        encoded = {}
        for name, encoder in self.modality_encoders.items():
            if name in observations:
                encoded[name] = encoder(observations[name])

        if self.fusion_type == "concat":
            features = torch.cat(list(encoded.values()), dim=-1)
            return self.output_activation(self.fusion(features))

        elif self.fusion_type == "attention":
            # Stack modalities as sequence
            projected = []
            for name in self.modalities:
                if name in encoded:
                    projected.append(self.modality_projs[name](encoded[name]))

            if not projected:
                return torch.zeros(1, device=next(self.parameters()).device)

            # [B, num_modalities, attention_dim]
            seq = torch.stack(projected, dim=1)

            # Self-attention
            attn_out, _ = self.attention(seq, seq, seq)

            # Pool (mean)
            pooled = attn_out.mean(dim=1)
            return self.output_activation(self.fusion(pooled))

        elif self.fusion_type == "film":
            primary = self.modalities[0]
            primary_feat = encoded[primary]

            other_feats = []
            for name in self.modalities[1:]:
                if name in encoded:
                    other_feats.append(encoded[name])

            if other_feats:
                other_concat = torch.cat(other_feats, dim=-1)
                gamma = self.gamma(primary_feat)
                beta = self.beta(primary_feat)
                modulated = gamma * other_concat + beta
                features = torch.cat([primary_feat, modulated], dim=-1)
            else:
                features = primary_feat

            return self.output_activation(self.fusion(features))


def create_encoder(config: Dict[str, Any], input_shape: Optional[tuple] = None) -> nn.Module:
    """Factory function to create encoder from config."""
    encoder_type = config.get("type", "mlp")

    if encoder_type == "mlp":
        return MLPEncoder(
            input_dim=config.get("input_dim", input_shape[0] if input_shape else 128),
            hidden_sizes=config.get("hidden_sizes", [128, 128]),
            activation=config.get("activation", "tanh"),
            layer_norm=config.get("layer_norm", False),
            output_size=config.get("output_size"),
            output_activation=config.get("output_activation"),
            dropout=config.get("dropout", 0.0),
            init_type=config.get("init_type", "orthogonal"),
            gain=config.get("gain", 1.0)
        )

    elif encoder_type == "cnn":
        return CNNEncoder(
            input_shape=config.get("input_shape", input_shape or (3, 64, 64)),
            channels=config.get("channels", [32, 64, 64]),
            kernels=config.get("kernels", [8, 4, 3]),
            strides=config.get("strides", [4, 2, 1]),
            paddings=config.get("paddings"),
            activation=config.get("activation", "relu"),
            layer_norm=config.get("layer_norm", False),
            output_size=config.get("output_size", 512),
            init_type=config.get("init_type", "orthogonal"),
            gain=config.get("gain", 1.0)
        )

    elif encoder_type == "rnn":
        return RNNEncoder(
            input_size=config.get("input_size", input_shape[0] if input_shape else 128),
            hidden_size=config.get("hidden_size", 256),
            num_layers=config.get("num_layers", 1),
            rnn_type=config.get("rnn_type", "lstm"),
            bidirectional=config.get("bidirectional", False),
            dropout=config.get("dropout", 0.0),
            batch_first=config.get("batch_first", True),
            init_type=config.get("init_type", "orthogonal"),
            gain=config.get("gain", 1.0)
        )

    elif encoder_type == "transformer":
        return TransformerEncoder(
            input_size=config.get("input_size", input_shape[0] if input_shape else 128),
            d_model=config.get("d_model", 256),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            dim_feedforward=config.get("dim_feedforward", 1024),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "gelu"),
            batch_first=config.get("batch_first", True),
            norm_first=config.get("norm_first", True),
            init_type=config.get("init_type", "orthogonal"),
            gain=config.get("gain", 1.0)
        )

    elif encoder_type == "lidar":
        return LidarEncoder(
            num_rays=config.get("num_rays", 360),
            input_channels=config.get("input_channels", 1),
            channels=config.get("channels", [32, 64, 128]),
            kernels=config.get("kernels", [5, 5, 3]),
            strides=config.get("strides", [2, 2, 1]),
            activation=config.get("activation", "relu"),
            pool=config.get("pool", "max"),
            output_size=config.get("output_size", 128),
            init_type=config.get("init_type", "orthogonal"),
            gain=config.get("gain", 1.0)
        )

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")