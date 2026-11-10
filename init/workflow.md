# Training Workflow Procedures

## Overview

This document defines the standard workflow for developing, training, and evaluating PPO agents in Webots. Follow these steps exactly.

---

## Phase 1: Environment Setup (One-time)

### 1.1 Install Dependencies
```bash
# Core ML
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gymnasium stable-baselines3  # For reference/comparison

# Webots
# Download from https://cyberbotics.com/ - install Webots R2023b+
# Set WEBOTS_HOME environment variable

# Project deps
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Pre-commit
pre-commit install
```

### 1.2 Verify Webots Integration
```bash
# Test Webots Python controller
cd webots/controllers/ppo_agent
python ppo_agent.py --test-connection

# Test world loading
webots --mode=fast --no-rendering webots/worlds/training.wbt
```

### 1.3 Validate Config System
```bash
python -m core.utils.validate_config config/ppo_config.yaml
python -m core.utils.validate_config agents/template/config.yaml
```

---

## Phase 2: Creating a New Agent

### 2.1 Use the Creation Script (MANDATORY)
```bash
# List available templates
python scripts/create_agent.py --list-templates

# Create agent from template
python scripts/create_agent.py \
    --name agent_003 \
    --template walker \
    --description "Bipedal walker on rough terrain" \
    --env webots/worlds/walker-training.wbt
```

### 2.2 Customize Agent Config
Edit `agents/agent_003/config.yaml`:
```yaml
# Inherit from global, override only what's needed
ppo:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5

neural:
  actor:
    hidden_sizes: [256, 256]
    activation: tanh
  critic:
    hidden_sizes: [256, 256]
    activation: tanh

env:
  world: webots/worlds/walker-training.wbt
  max_episode_steps: 1000
  observation_keys: [joint_pos, joint_vel, imu, lidar]
  action_keys: [motor_pos]
  reward_weights:
    forward_vel: 1.0
    energy: -0.01
    alive: 0.1

training:
  total_timesteps: 5_000_000
  eval_freq: 50_000
  save_freq: 100_000
  n_eval_episodes: 10
  seed: 42
```

### 2.3 Implement Custom Model (If Needed)
```python
# agents/agent_003/model/__init__.py
from core.neural import Actor, Critic, MLPEncoder
from core.neural.encoders import LidarEncoder

class CustomActor(Actor):
    def __init__(self, obs_dim, action_dim, config):
        # Custom encoder for lidar + proprioception
        self.lidar_encoder = LidarEncoder(config.lidar_channels)
        self.proprio_encoder = MLPEncoder(obs_dim - lidar_dim, config.hidden_sizes)
        # ... rest of implementation
```

### 2.4 Dry Run Validation
```bash
# Quick validation (1 episode, no saving)
python agents/agent_003/train.py --dry-run --verbose

# Check: no errors, logs created, metrics reasonable
```

---

## Phase 3: Training

### 3.1 Single Agent Training
```bash
# Standard training
python agents/agent_003/train.py

# Resume from checkpoint
python agents/agent_003/train.py --resume agents/agent_003/checkpoints/latest.pt

# Override config via CLI
python agents/agent_003/train.py --config-overrides ppo.learning_rate=1e-4 training.total_timesteps=1000000
```

### 3.2 Multi-Agent Orchestrated Training
```bash
# Train multiple agents with curriculum
python training/orchestrator.py \
    --config config/training_config.yaml \
    --agents agent_001,agent_002,agent_003 \
    --parallel 2 \
    --population-training
```

### 3.3 Monitoring Training
```bash
# TensorBoard
tensorboard --logdir agents/agent_003/logs/tensorboard

# Live metrics (terminal)
python scripts/visualize.py --agent agent_003 --live

# Compare agents
python scripts/visualize.py --compare agent_001,agent_002,agent_003 --metric episode_reward
```

### 3.4 Training Checklist (Per Run)
- [ ] Config validated
- [ ] Seed set and logged
- [ ] Git commit hash recorded
- [ ] TensorBoard logging active
- [ ] Checkpoints saving correctly
- [ ] Evaluation running at intervals
- [ ] No NaN losses/rewards
- [ ] KL divergence in expected range
- [ ] Entropy not collapsing too fast

---

## Phase 4: Evaluation

### 4.1 Standard Evaluation
```bash
# Evaluate latest checkpoint
python scripts/evaluate.py --agent agent_003 --episodes 100

# Evaluate specific checkpoint
python scripts/evaluate.py --agent agent_003 --checkpoint step_500000.pt --episodes 50

# Evaluate with video recording
python scripts/evaluate.py --agent agent_003 --episodes 10 --record-video
```

### 4.2 Evaluation Metrics (Required)
| Metric | Description | Target |
|--------|-------------|--------|
| Mean Episode Reward | Average return | > baseline |
| Success Rate | % episodes achieving goal | > 80% |
| Episode Length | Mean steps per episode | Stable |
| Action Smoothness | Mean action delta | Low |
| Energy Efficiency | Reward / energy cost | High |
| Generalization | Performance on eval world | > 70% of train |

### 4.3 Cross-Agent Comparison
```bash
# Generate comparison report
python scripts/visualize.py \
    --compare agent_001,agent_002,agent_003 \
    --output reports/comparison_20260115.html \
    --metrics episode_reward,success_rate,episode_length,energy
```

---

## Phase 5: Deployment/Transfer

### 5.1 Sim-to-Real Preparation
```bash
# Domain randomization config
python scripts/create_agent.py \
    --name agent_003_real \
    --template agent_003 \
    --domain-randomization \
    --output-dir agents/agent_003_real
```

### 5.2 Model Export
```bash
# Export to ONNX for deployment
python scripts/export_onnx.py \
    --agent agent_003 \
    --checkpoint best.pt \
    --output models/agent_003.onnx \
    --opset 17
```

---

## Emergency Procedures

### Training Divergence
1. **Stop training** (Ctrl+C)
2. **Check**: Learning rate too high? → Reduce by 10x
3. **Check**: Reward scale? → Normalize rewards
4. **Check**: Observation normalization? → Enable RunningMeanStd
5. **Resume** from last stable checkpoint

### Webots Crashes
1. Check `webots/controllers/ppo_agent/controller.log`
2. Common: Sensor noise → Add filtering in wrapper
3. Common: Physics instability → Reduce timestep in world
4. Restart Webots, resume from checkpoint

### OOM (Out of Memory)
1. Reduce `batch_size` in config
2. Reduce `n_steps` (rollout length)
3. Enable gradient checkpointing
4. Use `torch.compile()` (PyTorch 2.0+)

---

## Versioning & Releases

### Config Versioning
```yaml
# In every config.yaml
version: "1.2.0"  # SemVer
base_config: "config/ppo_config.yaml@v1.1.0"
```

### Release Checklist
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Configs validated
- [ ] Benchmark agents trained to convergence
- [ ] Comparison report generated
- [ ] Documentation updated
- [ ] Git tag: `v1.2.0-agent_003-walker`

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| New agent | `python scripts/create_agent.py --name agent_XXX --template TYPE` |
| Train | `python agents/agent_XXX/train.py` |
| Resume | `python agents/agent_XXX/train.py --resume PATH` |
| Evaluate | `python scripts/evaluate.py --agent agent_XXX --episodes 100` |
| Compare | `python scripts/visualize.py --compare agent_1,agent_2` |
| TensorBoard | `tensorboard --logdir agents/agent_XXX/logs/tensorboard` |
| Validate config | `python -m core.utils.validate_config PATH` |
| Export ONNX | `python scripts/export_onnx.py --agent agent_XXX` |
| Run tests | `pytest tests/ -v` |
| Lint | `ruff check . && black --check .` |

---

**REMEMBER**: Always read `init/rules.md` before making changes. The workflow is designed for reproducibility and collaboration - deviations require explicit approval.