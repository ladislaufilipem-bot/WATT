# PPO Neural Network Training Framework with Webots

A modular, efficient, and organized framework for training PPO (Proximal Policy Optimization) agents using Webots physics simulation. Each agent is isolated in its own directory with shared core libraries for maximum code reuse and maintainability.

## Features

- **Modular Architecture**: Each agent in its own directory (`agents/agent_XXX/`)
- **Config-Driven**: All hyperparameters in YAML configs with inheritance
- **Webots Integration**: Native Webots environment wrapper with supervisor communication
- **PPO Implementation**: Clean, efficient PPO with GAE, clipped loss, entropy bonus
- **Neural Networks**: Flexible architectures (MLP, CNN, RNN, Transformer, LiDAR encoders)
- **Multi-Agent Training**: Orchestrator for parallel agent training with curriculum learning
- **Reproducibility**: Full seeding, checkpointing, config versioning
- **Observability**: TensorBoard, Weights & Biases, structured JSONL logging

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Webots R2023b+** - Download from [cyberbotics.com](https://cyberbotics.com/)
3. **CUDA** (optional, for GPU training)

```bash
# Set Webots environment variable
export WEBOTS_HOME=/path/to/webots  # Linux/macOS
set WEBOTS_HOME=C:\Program Files\Webots  # Windows
```

### Installation

```bash
# Clone or navigate to project
cd ia

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Create Your First Agent

```bash
# List available templates
python scripts/create_agent.py --list

# Create a walker agent
python scripts/create_agent.py --name agent_001 --template walker --world walker-training.wbt

# Or create a custom agent
python scripts/create_agent.py --name agent_002 --template arm --description "6-DOF arm manipulation"
```

### Train an Agent

```bash
# Navigate to agent directory
cd agents/agent_001

# Quick test (1 episode)
python train.py --dry-run

# Full training
python train.py

# Resume from checkpoint
python train.py --resume checkpoints/latest.pt

# With Weights & Biases
python train.py --wandb --wandb-project my-project
```

### Evaluate an Agent

```bash
# Evaluate best checkpoint
python scripts/evaluate.py --agents agent_001 --episodes 50

# Compare multiple agents
python scripts/evaluate.py --agents agent_001,agent_002,agent_003 --compare --output comparison.json
```

### Visualize Training

```bash
# Plot single agent
python scripts/visualize.py --agents agent_001 --metric episode_reward

# Compare agents
python scripts/visualize.py --compare agent_001,agent_002 --metric episode_reward --output comparison.png

# HTML interactive report
python scripts/visualize.py --compare agent_001,agent_002 --format html
```

## Project Structure

```
ia/
├── CLAUDE.md              # Project guidance for Claude Code
├── init/                  # MANDATORY: Development rules (READ FIRST)
│   ├── rules.md           # Core rules (efficiency, organization, modularity)
│   ├── structure.md       # Project structure standards
│   └── workflow.md        # Training workflow procedures
├── config/                # Global configurations
│   ├── ppo_config.yaml    # PPO algorithm defaults
│   ├── neural_config.yaml # Neural network architectures
│   ├── webots_config.yaml # Webots simulation settings
│   └── training_config.yaml # Training pipeline settings
├── core/                  # Shared libraries (NO agent imports)
│   ├── ppo/               # PPO algorithm (agent, buffer, loss, update)
│   ├── neural/            # Networks (actor, critic, encoders)
│   ├── env/               # Environment wrappers (Gym, Webots)
│   └── utils/             # Utilities (config, seeding, logging, checkpoint, metrics)
├── agents/                # Individual agents (isolated)
│   ├── template/          # Template for new agents
│   ├── agent_001/         # Example: walker
│   └── agent_002/         # Example: arm
├── webots/                # Webots assets
│   ├── worlds/            # .wbt world files
│   ├── controllers/       # Robot controllers (Python)
│   ├── proto/             # PROTO definitions
│   └── assets/            # Meshes, textures, sounds
├── training/              # Orchestration
│   ├── orchestrator.py    # Multi-agent coordinator
│   ├── evaluator.py       # Evaluation utilities
│   └── curriculum.py      # Curriculum learning
├── scripts/               # CLI utilities
│   ├── create_agent.py    # Create agent from template
│   ├── run_training.py    # Run training pipeline
│   ├── evaluate.py        # Evaluate agents
│   └── visualize.py       # Plot logs, compare agents
├── tests/                 # Unit & integration tests
├── docs/                  # Documentation
├── pyproject.toml         # Project config
├── requirements.txt       # Core dependencies
├── requirements-dev.txt   # Dev dependencies
└── README.md              # This file
```

## Agent Structure

Each agent is completely self-contained:

```
agents/agent_XXX/
├── config.yaml          # Complete config (inherits from global + overrides)
├── train.py             # Training entry point
├── eval.py              # Evaluation entry point
├── model/
│   └── __init__.py      # Custom model definitions (optional)
├── logs/                # Training logs (JSONL, TensorBoard)
├── checkpoints/         # Model checkpoints
│   ├── best.pt          # Best evaluation performance
│   ├── latest.pt        # Most recent checkpoint
│   └── step_XXXXXX.pt   # Periodic checkpoints
└── videos/              # Evaluation videos (optional)
```

## Configuration

### Config Inheritance Hierarchy

```
config/ppo_config.yaml       # Base PPO defaults
    ↓
config/neural_config.yaml    # Base network defaults
    ↓
config/webots_config.yaml    # Base Webots defaults
    ↓
config/training_config.yaml  # Base training defaults
    ↓
agents/agent_XXX/config.yaml # Agent-specific OVERRIDES only
    ↓
CLI args (--key value)       # Runtime overrides
```

### Example Agent Config

```yaml
# agents/agent_001/config.yaml
version: "1.0.0"

agent:
  name: "agent_001"
  description: "Bipedal walker on flat terrain"

ppo:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01

neural:
  actor:
    hidden_sizes: [256, 256]
    activation: "tanh"
  critic:
    hidden_sizes: [256, 256]
    activation: "tanh"

env:
  world: "training.wbt"
  observation_keys: [joint_pos, joint_vel, imu, lidar]
  action_keys: [motor_pos]
  reward_weights:
    forward_vel: 1.0
    energy: -0.01
    alive: 0.1

training:
  total_timesteps: 5_000_000
  eval_freq: 50_000
  n_eval_episodes: 10
```

## Webots Integration

### World Setup

1. Create world in `webots/worlds/your_world.wbt`
2. Add robot with `DEF PPO_ROBOT` name
3. Add sensors (Lidar, IMU, GPS, Camera, etc.)
4. Add actuators (Motors with position control)
5. Add Supervisor for environment control

### Controller Communication

The framework uses Webots Emitter/Receiver for communication:

- **Supervisor → Controller**: Actions, reset commands, configuration
- **Controller → Supervisor**: Observations, rewards, done flags, info

### Custom Robot

Define your robot in `webots/proto/robots/YourRobot.proto`:

```proto
PROTO YourRobot [
  field SFVec3f translation 0 0 0
  field SFRotation rotation 0 0 1 0
] {
  Robot {
    name "PPO_ROBOT"
    controller "ppo_agent"
    children [
      # Your robot structure
    ]
  }
}
```

## Training Workflow

### 1. Environment Setup (One-time)
```bash
# Verify Webots integration
webots --mode=fast --no-rendering webots/worlds/training.wbt
python -m core.utils.validate_config config/ppo_config.yaml
```

### 2. Create Agent
```bash
python scripts/create_agent.py --name agent_003 --template walker --world walker-rough.wbt
```

### 3. Customize Reward Function
Edit `agents/agent_003/train.py`:
```python
def custom_reward_fn(obs, action, next_obs, info):
    reward = 0.0
    # Your reward logic here
    return reward, breakdown
```

### 4. Dry Run
```bash
python agents/agent_003/train.py --dry-run --verbose
```

### 5. Train
```bash
python agents/agent_003/train.py
# Monitor with: tensorboard --logdir agents/agent_003/logs/tensorboard
```

### 6. Evaluate
```bash
python scripts/evaluate.py --agent agent_003 --episodes 100 --output eval_results.json
```

## Multi-Agent Orchestration

```bash
# Train multiple agents with curriculum
python training/orchestrator.py \
    --config config/training_config.yaml \
    --agents agent_001,agent_002,agent_003 \
    --parallel 2
```

## Development Rules

**READ `init/rules.md` BEFORE MAKING CHANGES**

Key principles:
1. **Efficiency**: Reuse code in `core/`, configs over hardcoded values
2. **Organization**: One agent = one folder, shared code only in `core/`
3. **Modularity**: Single responsibility per module
4. **Reproducibility**: Versioned configs, seeded randomness, checkpointing
5. **Observability**: Structured logging, metrics, visualization

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html
```

## Model Export

```bash
# Export to ONNX for deployment
python scripts/export_onnx.py --agent agent_001 --checkpoint best.pt --output model.onnx
```

## Contributing

1. Read `init/rules.md` and `init/workflow.md`
2. Follow the config-driven approach
3. Add tests for new functionality
4. Run `ruff check . && black --check .` before committing
5. Update documentation as needed

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in research, please cite:
```bibtex
@software{ppo_webots,
  title = {PPO Neural Network Training Framework with Webots},
  year = {2024}
}
```