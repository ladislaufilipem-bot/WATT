# Project Structure Standards

## Directory Layout (Enforced)

```
ia/
├── CLAUDE.md              # Project guidance for Claude
├── init/                  # RULES - READ FIRST
│   ├── rules.md           # Core development rules
│   ├── structure.md       # This file
│   └── workflow.md        # Training workflow
├── config/                # GLOBAL CONFIGS (version controlled)
│   ├── ppo_config.yaml    # PPO algorithm defaults
│   ├── webots_config.yaml # Webots connection/simulation
│   ├── training_config.yaml # Orchestration settings
│   └── neural_config.yaml # Network architecture defaults
├── core/                  # SHARED LIBRARY (no agent imports)
│   ├── __init__.py
│   ├── ppo/
│   │   ├── __init__.py
│   │   ├── agent.py       # PPOAgent class
│   │   ├── buffer.py      # RolloutBuffer
│   │   ├── loss.py        # PPO loss functions
│   │   └── update.py      # Update step logic
│   ├── neural/
│   │   ├── __init__.py
│   │   ├── actor.py       # Actor network
│   │   ├── critic.py      # Critic network
│   │   ├── encoders.py    # Observation encoders (CNN, MLP, RNN)
│   │   └── utils.py       # Weight init, activations
│   ├── env/
│   │   ├── __init__.py
│   │   ├── base.py        # BaseEnvWrapper (Gym interface)
│   │   ├── webots.py      # WebotsEnvWrapper
│   │   ├── vec_env.py     # Vectorized envs
│   │   └── wrappers.py    # Normalization, frame stack, etc.
│   └── utils/
│       ├── __init__.py
│       ├── logging.py     # Structured logging setup
│       ├── seeding.py     # Global seeding
│       ├── metrics.py     # Metrics computation
│       ├── checkpoint.py  # Save/load checkpoints
│       └── config.py      # Config loading/validation
├── agents/                # AGENT DIRECTORIES (isolated)
│   ├── template/          # TEMPLATE - copy via script only
│   │   ├── config.yaml    # Template config
│   │   ├── train.py       # Template training script
│   │   ├── eval.py        # Template eval script
│   │   ├── model/
│   │   │   └── __init__.py
│   │   ├── logs/
│   │   └── checkpoints/
│   ├── agent_001/         # Example: Walker
│   │   ├── config.yaml
│   │   ├── train.py
│   │   ├── eval.py
│   │   ├── model/
│   │   ├── logs/
│   │   └── checkpoints/
│   └── agent_002/         # Example: Arm
│       └── ...
├── webots/                # WEBOTS ASSETS
│   ├── worlds/
│   │   ├── training.wbt       # Main training world
│   │   ├── evaluation.wbt     # Evaluation world
│   │   └── curriculum/        # Curriculum worlds
│   ├── controllers/
│   │   ├── ppo_agent/         # Agent controller (receives actions)
│   │   │   ├── ppo_agent.py
│   │   │   └── __init__.py
│   │   ├── supervisor/        # Supervisor controller
│   │   │   ├── supervisor.py
│   │   │   └── __init__.py
│   │   └── random/            # Random policy for debugging
│   │       └── ...
│   ├── proto/
│   │   ├── robots/            # Robot PROTOs
│   │   ├── sensors/           # Sensor PROTOs
│   │   └── environments/      # Environment PROTOs
│   └── assets/
│       ├── textures/
│       ├── meshes/
│       └── sounds/
├── training/              # ORCHESTRATION
│   ├── __init__.py
│   ├── orchestrator.py    # Multi-agent training coordinator
│   ├── evaluator.py       # Cross-agent evaluation
│   ├── curriculum.py      # Curriculum learning scheduler
│   └── population.py      # Population-based training
├── scripts/               # CLI UTILITIES
│   ├── create_agent.py    # Create agent from template
│   ├── run_training.py    # Run training (single/multi)
│   ├── evaluate.py        # Evaluate agent(s)
│   ├── visualize.py       # Plot logs, compare agents
│   └── migrate.py         # Migrate agents across versions
├── tests/                 # TESTS
│   ├── unit/
│   │   ├── test_ppo.py
│   │   ├── test_neural.py
│   │   ├── test_env.py
│   │   └── test_utils.py
│   ├── integration/
│   │   ├── test_training_loop.py
│   │   └── test_webots_integration.py
│   └── fixtures/
│       └── dummy_config.yaml
├── docs/                  # DOCUMENTATION
│   ├── architecture.md
│   ├── api_reference.md
│   ├── webots_setup.md
│   └── troubleshooting.md
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Agent Directory Structure (Mandatory)

Each agent **MUST** have:
```
agents/agent_XXX/
├── config.yaml          # Complete config (inherits from global + overrides)
├── train.py             # Entry point: python train.py [--resume]
├── eval.py              # Evaluation entry point
├── model/
│   └── __init__.py      # Model definition (can customize architecture)
├── logs/
│   ├── train_YYYYMMDD_HHMMSS.jsonl
│   └── eval_YYYYMMDD_HHMMSS.jsonl
├── checkpoints/
│   ├── best.pt
│   ├── latest.pt
│   └── step_XXXXXX.pt
└── videos/              # Optional: evaluation videos
    └── eval_ep_XXX.mp4
```

## Config Inheritance Hierarchy

```
config/ppo_config.yaml       # Base PPO defaults
    ↓
config/neural_config.yaml    # Base network defaults
    ↓
config/webots_config.yaml    # Base Webots defaults
    ↓
agents/agent_XXX/config.yaml # Agent-specific OVERRIDES only
    ↓
CLI args (--key value)       # Runtime overrides
```

## Naming Conventions

| Item | Convention | Example |
|------|------------|---------|
| Agent dir | `agent_XXX` (zero-padded) | `agent_001`, `agent_042` |
| Config keys | `snake_case` | `learning_rate`, `gae_lambda` |
| Python modules | `snake_case` | `rollout_buffer.py` |
| Classes | `PascalCase` | `PPOAgent`, `RolloutBuffer` |
| Functions | `snake_case` | `compute_gae()`, `update_policy()` |
| Constants | `UPPER_SNAKE` | `MAX_EPISODE_STEPS` |
| Checkpoints | `best.pt`, `latest.pt`, `step_N.pt` | `step_100000.pt` |
| Log files | `train_YYYYMMDD_HHMMSS.jsonl` | `train_20260115_143022.jsonl` |
| Webots worlds | `kebab-case.wbt` | `walker-training.wbt` |
| PROTO names | `PascalCase` | `WalkerRobot`, `LidarSensor` |

## Version Control Rules

- **Track**: All configs, core code, scripts, tests, docs, Webots worlds/controllers/PROTOs
- **Ignore**: `agents/*/logs/`, `agents/*/checkpoints/`, `agents/*/videos/`, `agents/*/model/__pycache__/`, `*.pyc`, `.pytest_cache/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`
- **LFS**: Large assets (meshes, textures, videos) → Git LFS