# Development Rules - MANDATORY READ

**Read this file before ANY code changes. These rules are non-negotiable.**

## Core Principles

### 1. EFFICIENCY (Token & Compute Optimization)
- **Reuse over rewrite**: Check `core/` before creating new functionality
- **Config over code**: All hyperparameters in YAML, no magic numbers in Python
- **Batch operations**: Vectorized environments, parallel training
- **Checkpoint strategically**: Save only best/last, not every epoch
- **Lazy imports**: Import heavy libs (torch, webots) only when needed

### 2. ORGANIZATION (Structure & Isolation)
- **One agent = one folder** under `agents/` with complete self-containment
- **Shared code ONLY in `core/`** - zero cross-agent imports
- **Config hierarchy**: Global → Agent → Override (CLI args)
- **Naming convention**: `agent_XXX` (zero-padded), descriptive subdirs
- **No global state**: Each module stateless, pass config explicitly

### 3. MODULARITY (Single Responsibility)
```
core/
├── ppo/          # ONLY PPO algorithm (loss, update, buffer)
├── neural/       # ONLY network architectures (Actor, Critic, Encoders)
├── env/          # ONLY environment wrappers (Gym/Webots interface)
└── utils/        # ONLY pure utilities (logging, seeding, metrics)
```

### 4. REPRODUCIBILITY (Science-First)
- **Seed everything**: `torch`, `numpy`, `random`, environment
- **Version configs**: Git-track all YAML configs
- **Deterministic ops**: `torch.use_deterministic_algorithms(True)`
- **Checkpoint format**: `{model_state, optimizer_state, config, step, metrics}`

### 5. OBSERVABILITY (Debugging & Analysis)
- **Structured logging**: JSONL format, standardized fields
- **TensorBoard/Weights&Biases**: Auto-log losses, rewards, grads, params
- **Video recording**: Periodic eval episodes saved as MP4
- **Metrics dashboard**: Episode reward, length, entropy, KL, clipfrac

## File Standards

### Python
- Type hints mandatory (`from __future__ import annotations`)
- Docstrings: Google style for public APIs
- Max line length: 100 chars
- Ruff + Black formatted

### YAML Configs
- Snake_case keys
- Comments for non-obvious params
- No inline secrets (use env vars)

### Webots
- Worlds: `.wbt` in `webots/worlds/`
- Controllers: Python in `webots/controllers/<name>/`
- PROTOs: `webots/proto/` for reusable robot definitions

## Workflow Rules

### Creating New Agent
```bash
# ONLY via script - never manual copy
python scripts/create_agent.py --name agent_XXX --template <type>
```

### Training
1. **Validate config** → `python -m core.utils.validate_config agents/agent_XXX/config.yaml`
2. **Dry run** (1 episode) → `python agents/agent_XXX/train.py --dry-run`
3. **Full training** → `python agents/agent_XXX/train.py`
4. **Evaluate** → `python scripts/evaluate.py --agent agent_XXX`

### Modifying Core
- **Run tests first**: `pytest tests/core/ -v`
- **No breaking changes** without version bump in `config/ppo_config.yaml`
- **Update ALL agents** if interface changes (use orchestrator)

## Prohibited Patterns

| Pattern | Why | Alternative |
|---------|-----|-------------|
| `import agents.agent_001` in core | Coupling | Pass config/data explicitly |
| Hardcoded paths | Portability | Config-driven paths |
| Global variables | State leaks | Dependency injection |
| `print()` for logging | No structure | `logging` + structured JSONL |
| Magic numbers | Irreproducibility | YAML config with comments |
| Direct Webots API in agents | Sim coupling | `core.env.WebotsWrapper` |

## Enforcement

- **Pre-commit hooks**: Ruff, Black, MyPy, config validation
- **CI pipeline**: Tests + lint + config validation on every PR
- **Code review**: Checklist includes rules compliance

---

**VIOLATION CONSEQUENCE**: Changes violating these rules will be rejected. Request rule modification via explicit user approval with rationale.