# Custom Model for Template Agent
# Override this file to customize network architecture

from core.neural.actor import create_actor
from core.neural.critic import create_critic
from core.neural.encoders import create_encoder


def create_custom_actor(config, obs_dim, action_dim):
    """Create custom actor - override for custom architectures."""
    # Example: Add custom encoder
    # config["encoder"] = {"type": "lidar", "num_rays": 360, "output_size": 128}
    return create_actor(config, obs_dim, action_dim)


def create_custom_critic(config, obs_dim, action_dim):
    """Create custom critic - override for custom architectures."""
    # Example: Use twin critic
    # config["type"] = "twin"
    return create_critic(config, obs_dim, action_dim)


# Example custom encoder for LiDAR + proprioception
class CustomWalkerEncoder(torch.nn.Module):
    """Custom encoder combining LiDAR and proprioception."""

    def __init__(self, config):
        super().__init__()

        from core.neural.encoders import LidarEncoder, MLPEncoder
        import torch.nn as nn

        # LiDAR encoder
        self.lidar_encoder = LidarEncoder(
            num_rays=config.get("lidar_rays", 360),
            output_size=config.get("lidar_features", 128)
        )

        # Proprioception encoder
        proprio_dim = config.get("proprio_dim", 27)  # joint_pos + joint_vel + imu
        self.proprio_encoder = MLPEncoder(
            input_dim=proprio_dim,
            hidden_sizes=config.get("proprio_hidden", [128, 128]),
            output_size=config.get("proprio_features", 128)
        )

        # Fusion
        fusion_size = config.get("lidar_features", 128) + config.get("proprio_features", 128)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.output_size = 256

    def forward(self, obs_dict):
        """Forward pass with dict observations."""
        lidar = obs_dict.get("lidar")
        proprio = torch.cat([
            obs_dict.get("joint_pos", torch.zeros(1)),
            obs_dict.get("joint_vel", torch.zeros(1)),
            obs_dict.get("imu", torch.zeros(1))
        ], dim=-1)

        lidar_features = self.lidar_encoder(lidar)
        proprio_features = self.proprio_encoder(proprio)

        combined = torch.cat([lidar_features, proprio_features], dim=-1)
        return self.fusion(combined)


# Example usage in train.py:
# from model import CustomWalkerEncoder
#
# # In agent creation, replace encoder creation:
# encoder = CustomWalkerEncoder(config.get("encoder", {}))