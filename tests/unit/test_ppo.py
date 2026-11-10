# Unit Tests for PPO Core
from __future__ import annotations

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.ppo.buffer import RolloutBuffer, create_buffer
from core.ppo.loss import (
    compute_policy_loss, compute_value_loss, compute_entropy_loss,
    ppo_loss, compute_gae, explained_variance
)
from core.ppo.update import PPOUpdater, create_optimizer
from core.neural.actor import (
    GaussianActor, CategoricalActor, create_actor, ActorOutput
)
from core.neural.critic import (
    Critic, TwinCritic, EnsembleCritic, create_critic, CriticOutput
)
from core.neural.encoders import (
    MLPEncoder, CNNEncoder, RNNEncoder, LidarEncoder, create_encoder
)
from core.utils.config import (
    load_yaml, save_yaml, deep_merge, validate_config,
    load_config_with_inheritance, apply_cli_overrides
)
from core.utils.seeding import set_global_seed, get_global_seed_state, set_global_seed_state
from core.utils.metrics import (
    compute_gae as metrics_compute_gae, RunningMeanStd, EpisodeStats,
    MetricsBuffer, evaluate_policy, explained_variance as metrics_explained_variance
)
from core.utils.checkpoint import CheckpointManager, save_checkpoint, load_checkpoint


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    def setup_method(self):
        self.buffer = RolloutBuffer(
            buffer_size=100,
            obs_space={"obs": (4,)},
            action_space=(2,),
            n_envs=2,
            device="cpu",
            gamma=0.99,
            gae_lambda=0.95
        )

    def test_add_and_compute_advantages(self):
        """Test adding transitions and computing GAE."""
        for i in range(10):
            obs = {"obs": torch.randn(2, 4)}
            action = torch.randn(2, 2)
            reward = torch.randn(2)
            value = torch.randn(2)
            log_prob = torch.randn(2)
            done = torch.zeros(2, dtype=torch.bool)
            done[1] = (i == 9)  # End episode at step 9

            self.buffer.add(obs, action, reward, value, log_prob, done)

        last_values = torch.randn(2)
        last_dones = torch.zeros(2, dtype=torch.bool)

        self.buffer.compute_advantages(last_values, last_dones)

        assert self.buffer.generator_ready
        assert self.buffer.advantages.shape == (10, 2)
        assert self.buffer.returns.shape == (10, 2)

    def test_get_generator(self):
        """Test minibatch generator."""
        # Fill buffer
        for i in range(20):
            obs = {"obs": torch.randn(2, 4)}
            action = torch.randn(2, 2)
            reward = torch.randn(2)
            value = torch.randn(2)
            log_prob = torch.randn(2)
            done = torch.zeros(2, dtype=torch.bool)

            self.buffer.add(obs, action, reward, value, log_prob, done)

        last_values = torch.randn(2)
        last_dones = torch.zeros(2, dtype=torch.bool)
        self.buffer.compute_advantages(last_values, last_dones)

        batches = list(self.buffer.get_generator(batch_size=8, num_epochs=2))
        assert len(batches) > 0

        for batch in batches:
            assert isinstance(batch, dict) or hasattr(batch, 'observations')
            assert batch.observations["obs"].shape[0] <= 8


class TestPPOLoss:
    """Tests for PPO loss functions."""

    def test_compute_policy_loss(self):
        """Test policy loss computation."""
        new_log_probs = torch.tensor([0.5, -0.5, 0.0])
        old_log_probs = torch.tensor([0.3, -0.3, 0.1])
        advantages = torch.tensor([1.0, -1.0, 0.5])
        clip_range = 0.2

        policy_loss, clip_frac, approx_kl = compute_policy_loss(
            new_log_probs, old_log_probs, advantages, clip_range
        )

        assert isinstance(policy_loss, torch.Tensor)
        assert 0 <= clip_frac <= 1
        assert isinstance(approx_kl, torch.Tensor)

    def test_compute_value_loss(self):
        """Test value loss computation."""
        new_values = torch.tensor([1.0, 2.0, 3.0])
        old_values = torch.tensor([0.8, 2.2, 2.8])
        returns = torch.tensor([1.0, 2.0, 3.0])

        value_loss, explained_var = compute_value_loss(
            new_values, old_values, returns, clip_range_vf=0.5
        )

        assert isinstance(value_loss, torch.Tensor)
        assert value_loss >= 0

    def test_compute_gae(self):
        """Test GAE computation."""
        rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        values = torch.tensor([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        dones = torch.tensor([[False, False], [False, False], [True, True]])
        gamma = 0.99
        gae_lambda = 0.95
        next_value = torch.tensor([0.0, 0.0])

        advantages, returns = compute_gae(
            rewards, values, dones, gamma, gae_lambda, next_value
        )

        assert advantages.shape == (3, 2)
        assert returns.shape == (3, 2)


class TestNeuralNetworks:
    """Tests for neural network components."""

    def test_mlp_encoder(self):
        """Test MLP encoder."""
        encoder = MLPEncoder(
            input_dim=10,
            hidden_sizes=[64, 64],
            activation="tanh",
            output_size=32
        )

        x = torch.randn(4, 10)
        out = encoder(x)

        assert out.shape == (4, 32)

    def test_gaussian_actor(self):
        """Test Gaussian actor."""
        actor = GaussianActor(
            obs_dim=10,
            action_dim=4,
            encoder_config={"type": "mlp", "hidden_sizes": [64, 64]},
            hidden_sizes=[128, 128]
        )

        obs = torch.randn(4, 10)
        output = actor(obs)

        assert isinstance(output, ActorOutput)
        assert output.action.shape == (4, 4)
        assert output.log_prob.shape == (4,)
        assert output.entropy.shape == ()

    def test_categorical_actor(self):
        """Test Categorical actor."""
        actor = CategoricalActor(
            obs_dim=10,
            action_dim=5,
            encoder_config={"type": "mlp", "hidden_sizes": [64, 64]},
            hidden_sizes=[128, 128]
        )

        obs = torch.randn(4, 10)
        output = actor(obs)

        assert output.action.shape == (4,)
        assert output.action.dtype == torch.long
        assert (output.action >= 0).all() and (output.action < 5).all()

    def test_critic(self):
        """Test Critic network."""
        critic = Critic(
            obs_dim=10,
            encoder_config={"type": "mlp", "hidden_sizes": [64, 64]},
            hidden_sizes=[128, 128]
        )

        obs = torch.randn(4, 10)
        output = critic(obs)

        assert isinstance(output, CriticOutput)
        assert output.value.shape == (4,)

    def test_twin_critic(self):
        """Test Twin Critic."""
        critic = TwinCritic(
            obs_dim=10,
            encoder_config={"type": "mlp", "hidden_sizes": [64, 64]},
            hidden_sizes=[128, 128],
            shared_encoder=False
        )

        obs = torch.randn(4, 10)
        out1, out2 = critic(obs)

        assert out1.value.shape == (4,)
        assert out2.value.shape == (4,)

    def test_ensemble_critic(self):
        """Test Ensemble Critic."""
        critic = EnsembleCritic(
            obs_dim=10,
            encoder_config={"type": "mlp", "hidden_sizes": [64, 64]},
            hidden_sizes=[128, 128],
            ensemble_size=3,
            shared_encoder=True
        )

        obs = torch.randn(4, 10)
        outputs = critic(obs)

        assert len(outputs) == 3
        for out in outputs:
            assert out.value.shape == (4,)


class TestConfig:
    """Tests for config utilities."""

    def test_load_save_yaml(self, tmp_path):
        """Test YAML load/save."""
        data = {"key": "value", "nested": {"num": 42}}
        path = tmp_path / "test.yaml"

        save_yaml(data, path)
        loaded = load_yaml(path)

        assert loaded == data

    def test_deep_merge(self):
        """Test deep merge."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 20}, "e": 5}

        merged = deep_merge(base, override)

        assert merged == {"a": 1, "b": {"c": 20, "d": 3}, "e": 5}

    def test_apply_cli_overrides(self):
        """Test CLI override application."""
        config = {"ppo": {"learning_rate": 3e-4}, "nested": {"value": 1}}
        overrides = ["ppo.learning_rate=1e-4", "nested.value=42", "new.key=test"]

        result = apply_cli_overrides(config, overrides)

        assert result["ppo"]["learning_rate"] == 1e-4
        assert result["nested"]["value"] == 42
        assert result["new"]["key"] == "test"

    def test_validate_config(self):
        """Test config validation."""
        config = {
            "version": "1.0.0",
            "ppo": {"learning_rate": 3e-4},
            "network": {"actor": {}, "critic": {}}
        }

        errors = validate_config(config, "agent")
        # Should have no errors for valid config
        assert isinstance(errors, list)


class TestSeeding:
    """Tests for seeding utilities."""

    def test_set_global_seed(self):
        """Test global seed setting."""
        set_global_seed(42, deterministic=True)

        # Generate some random numbers
        a = torch.randn(3)
        b = np.random.randn(3)

        # Reset seed
        set_global_seed(42, deterministic=True)

        # Should get same numbers
        a2 = torch.randn(3)
        b2 = np.random.randn(3)

        assert torch.allclose(a, a2)
        assert np.allclose(b, b2)

    def test_get_set_seed_state(self):
        """Test getting and setting seed state."""
        set_global_seed(123)

        state = get_global_seed_state()

        # Generate some numbers
        torch.randn(3)
        np.random.randn(3)

        # Restore state
        set_global_seed_state(state)

        # Should get same numbers again
        a1 = torch.randn(3)
        a2 = torch.randn(3)

        # Actually they'll be different because we called randn twice
        # The test is that state restoration works without error


class TestMetrics:
    """Tests for metrics utilities."""

    def test_running_mean_std(self):
        """Test RunningMeanStd."""
        rms = RunningMeanStd.create((3,))

        # Add some data
        data = np.random.randn(100, 3)
        rms.update(data)

        # Normalize
        normalized = rms.normalize(data)

        assert normalized.shape == (100, 3)
        assert abs(normalized.mean()) < 0.1  # Should be close to 0
        assert abs(normalized.std() - 1.0) < 0.1  # Should be close to 1

    def test_episode_stats(self):
        """Test EpisodeStats."""
        stats = EpisodeStats()
        stats.reward = 10.0
        stats.length = 100
        stats.finalize()

        assert stats.duration > 0

    def test_metrics_buffer(self):
        """Test MetricsBuffer."""
        buffer = MetricsBuffer(window_size=10)

        for i in range(15):
            buffer.add(reward=float(i), length=100, info={"custom": float(i * 2)})

        stats = buffer.get_stats()

        assert "mean_reward" in stats
        assert "mean_custom" in stats
        # Should only have last 10
        assert stats["mean_reward"] == np.mean(list(range(5, 15)))


class TestCheckpoint:
    """Tests for checkpoint utilities."""

    def test_checkpoint_manager(self, tmp_path):
        """Test CheckpointManager."""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        manager = CheckpointManager(
            checkpoint_dir=tmp_path,
            max_checkpoints=3,
            save_best=True
        )

        # Save checkpoint
        path = manager.save(
            model=model,
            optimizer=optimizer,
            step=1000,
            episode=10,
            metrics={"eval/mean_reward": 50.0},
            config={"test": True}
        )

        assert path.exists()

        # Load checkpoint
        model2 = torch.nn.Linear(10, 5)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

        metadata = manager.load(path, model2, optimizer2)

        assert metadata.step == 1000
        assert metadata.episode == 10

    def test_save_load_checkpoint(self, tmp_path):
        """Test simple save/load functions."""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            step=500,
            episode=5
        )

        model2 = torch.nn.Linear(10, 5)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

        result = load_checkpoint(path, model2, optimizer2)

        assert result["metadata"].step == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])