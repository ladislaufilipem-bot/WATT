# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PPO Neural Network Training Framework with Webots Simulation**

This project implements a modular PPO (Proximal Policy Optimization) training system where each AI agent is isolated in its own directory, with physical simulation training conducted in Webots.

## Architecture

```
ia/
├── CLAUDE.md              # This file - project guidance
├── init/                  # MANDATORY: Read first - contains development rules
│   ├── rules.md           # Core development rules (efficiency, organization)
│   ├── structure.md       # Project structure standards
│   └── workflow.md        # Training workflow procedures
├── config/                # Global configuration
│   ├── ppo_config.yaml    # PPO hyperparameters
│   ├── webots_config.yaml # Webots simulation settings
│   └── training_config.yaml # Training pipeline settings
├── core/                  # Shared core libraries
│   ├── ppo/               # PPO algorithm implementation
│   ├── neural/            # Neural network architectures
│   ├── env/               # Environment wrappers
│   └── utils/             # Shared utilities
├── agents/                # Individual AI agents (each in own folder)
│   ├── agent_001/         # Example agent structure
│   │   ├── config.yaml    # Agent-specific config
│   │   ├── model/         # Saved models
│   │   ├── logs/          # Training logs
│   │   ├── checkpoints/   # Model checkpoints
│   │   └── train.py       # Agent training entry point
│   └── template/          # Template for new agents
├── webots/                # Webots integration
│   ├── worlds/            # Webots world files (.wbt)
│   ├── controllers/       # Robot controllers
│   ├── proto/             # PROTO definitions
│   └── supervisor/        # Supervisor controllers
├── training/              # Training orchestration
│   ├── orchestrator.py    # Multi-agent training coordinator
│   ├── evaluator.py       # Agent evaluation
│   └── curriculum.py      # Curriculum learning
├── scripts/               # Utility scripts
│   ├── create_agent.py    # Create new agent from template
│   ├── run_training.py    # Run training pipeline
│   └── evaluate.py        # Evaluate trained agents
└── tests/                 # Unit and integration tests
```

## Development Rules (MANDATORY - Read init/rules.md First)

1. **Efficiency**: Minimize token usage, reuse code, avoid duplication
2. **Organization**: Each agent isolated, shared code in `core/`, configs in `config/`
3. **Modularity**: Single responsibility per module, clear interfaces
4. **Reproducibility**: Versioned configs, seeded randomness, checkpointing
5. **Observability**: Comprehensive logging, metrics, visualization

## Commands

```bash
# Create new agent from template
python scripts/create_agent.py --name agent_002 --template walker

# Run single agent training
python agents/agent_001/train.py --config agents/agent_001/config.yaml

# Run multi-agent orchestrated training
python training/orchestrator.py --config config/training_config.yaml

# Evaluate agent
python scripts/evaluate.py --agent agent_001 --episodes 100

# Run Webots simulation (headless)
webots --mode=fast --no-rendering webots/worlds/training.wbt

# Run tests
pytest tests/ -v

# Lint
ruff check .
black --check .
```

## Key Principles

- **Never modify `init/` rules without explicit user approval**
- **Always read `init/rules.md` before making changes**
- **Each agent is independent** - own config, model, logs
- **Shared code only in `core/`** - no cross-agent dependencies
- **Webots worlds/controllers in `webots/`** - version controlled
- **Config-driven** - YAML configs, no hardcoded values