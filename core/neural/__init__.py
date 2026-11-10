# Neural Networks Package
"""
Neural Network Architectures for PPO
"""

from core.neural.actor import (
    create_actor, GaussianActor, SquashedGaussianActor,
    CategoricalActor, MixtureActor, ActorOutput
)
from core.neural.critic import (
    create_critic, Critic, TwinCritic, EnsembleCritic,
    StateActionCritic, CriticOutput
)
from core.neural.encoders import (
    create_encoder, MLPEncoder, CNNEncoder, RNNEncoder,
    TransformerEncoder, LidarEncoder, MultiModalFusion,
    initialize_weights, get_activation
)

__all__ = [
    # Actor
    "create_actor", "GaussianActor", "SquashedGaussianActor",
    "CategoricalActor", "MixtureActor", "ActorOutput",
    # Critic
    "create_critic", "Critic", "TwinCritic", "EnsembleCritic",
    "StateActionCritic", "CriticOutput",
    # Encoders
    "create_encoder", "MLPEncoder", "CNNEncoder", "RNNEncoder",
    "TransformerEncoder", "LidarEncoder", "MultiModalFusion",
    "initialize_weights", "get_activation",
]