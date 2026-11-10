# Configuration Loading and Validation Utilities
from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from copy import deepcopy

import jsonschema
from jsonschema import validate


@dataclass
class ConfigSchema:
    """Schema definition for config validation."""
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    additionalProperties: bool = True


# Base schemas for each config type
PPO_SCHEMA = ConfigSchema(
    type="object",
    properties={
        "version": {"type": "string"},
        "ppo": {"type": "object"},
        "optimizer": {"type": "object"},
        "lr_schedule": {"type": "object"},
        "network": {"type": "object"},
        "preprocessing": {"type": "object"},
        "exploration": {"type": "object"},
        "distributed": {"type": "object"},
        "amp": {"type": "object"},
        "logging": {"type": "object"},
    },
    required=["version", "ppo", "network"]
)

WEBOTS_SCHEMA = ConfigSchema(
    type="object",
    properties={
        "version": {"type": "string"},
        "webots": {"type": "object"},
        "simulation": {"type": "object"},
        "robot": {"type": "object"},
        "communication": {"type": "object"},
        "spaces": {"type": "object"},
        "domain_randomization": {"type": "object"},
        "curriculum": {"type": "object"},
        "performance": {"type": "object"},
    },
    required=["version", "webots", "simulation", "robot"]
)

TRAINING_SCHEMA = ConfigSchema(
    type="object",
    properties={
        "version": {"type": "string"},
        "training": {"type": "object"},
        "orchestration": {"type": "object"},
        "resources": {"type": "object"},
        "monitoring": {"type": "object"},
        "checkpointing": {"type": "object"},
        "early_stopping": {"type": "object"},
        "recovery": {"type": "object"},
        "experiment": {"type": "object"},
    },
    required=["version", "training"]
)

NEURAL_SCHEMA = ConfigSchema(
    type="object",
    properties={
        "version": {"type": "string"},
        "network": {"type": "object"},
        "encoders": {"type": "object"},
        "actor": {"type": "object"},
        "critic": {"type": "object"},
        "shared_encoder": {"type": "object"},
        "advanced": {"type": "object"},
        "regularization": {"type": "object"},
        "export": {"type": "object"},
    },
    required=["version", "network", "encoders", "actor", "critic"]
)

AGENT_SCHEMA = ConfigSchema(
    type="object",
    properties={
        "version": {"type": "string"},
        "agent": {"type": "object"},
        "ppo": {"type": "object"},
        "neural": {"type": "object"},
        "env": {"type": "object"},
        "training": {"type": "object"},
    },
    required=["version", "agent"]
)

SCHEMAS = {
    "ppo": PPO_SCHEMA,
    "webots": WEBOTS_SCHEMA,
    "training": TRAINING_SCHEMA,
    "neural": NEURAL_SCHEMA,
    "agent": AGENT_SCHEMA,
}


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file safely."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save YAML file with consistent formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries. Override takes precedence."""
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_config_with_inheritance(
    agent_config_path: Union[str, Path],
    global_configs: Optional[Dict[str, Union[str, Path]]] = None
) -> Dict[str, Any]:
    """
    Load agent config with global config inheritance.

    Hierarchy (lowest to highest priority):
    1. config/ppo_config.yaml
    2. config/neural_config.yaml
    3. config/webots_config.yaml
    4. config/training_config.yaml
    5. agents/agent_XXX/config.yaml
    6. CLI overrides (handled separately)
    """
    agent_path = Path(agent_config_path)
    agent_config = load_yaml(agent_path)

    # Default global config paths
    if global_configs is None:
        base_dir = agent_path.parent.parent.parent  # ia/
        global_configs = {
            "ppo": base_dir / "config" / "ppo_config.yaml",
            "neural": base_dir / "config" / "neural_config.yaml",
            "webots": base_dir / "config" / "webots_config.yaml",
            "training": base_dir / "config" / "training_config.yaml",
        }

    # Load and merge global configs
    merged = {}
    for name, path in global_configs.items():
        if Path(path).exists():
            global_config = load_yaml(path)
            merged = deep_merge(merged, global_config)

    # Merge agent config (overrides)
    final_config = deep_merge(merged, agent_config)

    return final_config


def validate_config(config: Dict[str, Any], schema_name: str) -> List[str]:
    """Validate config against schema. Returns list of errors (empty if valid)."""
    if schema_name not in SCHEMAS:
        return [f"Unknown schema: {schema_name}"]

    schema = SCHEMAS[schema_name]
    errors = []

    try:
        validate(instance=config, schema=schema.__dict__)
    except jsonschema.ValidationError as e:
        errors.append(f"{schema_name}: {e.message} at {' -> '.join(str(p) for p in e.path)}")
    except Exception as e:
        errors.append(f"{schema_name}: Validation error - {e}")

    return errors


def validate_all_configs(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate all config sections. Returns dict of section -> errors."""
    results = {}

    # Validate each section if present
    if "ppo" in config:
        results["ppo"] = validate_config(config, "ppo")
    if "webots" in config or "simulation" in config:
        results["webots"] = validate_config(config, "webots")
    if "training" in config:
        results["training"] = validate_config(config, "training")
    if "network" in config or "encoders" in config:
        results["neural"] = validate_config(config, "neural")

    return results


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get nested config value using dot notation (e.g., 'ppo.learning_rate')."""
    keys = key.split('.')
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def set_config_value(config: Dict[str, Any], key: str, value: Any) -> None:
    """Set nested config value using dot notation."""
    keys = key.split('.')
    target = config

    for k in keys[:-1]:
        if k not in target:
            target[k] = {}
        target = target[k]

    target[keys[-1]] = value


def apply_cli_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply CLI overrides in format 'key=value'."""
    result = deepcopy(config)

    for override in overrides:
        if '=' not in override:
            continue
        key, value_str = override.split('=', 1)

        # Parse value type
        try:
            # Try JSON parsing for complex types
            import json
            value = json.loads(value_str)
        except json.JSONDecodeError:
            # Fallback to string
            value = value_str

        set_config_value(result, key, value)

    return result


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic hash of config for reproducibility."""
    import hashlib
    import json

    # Sort keys for deterministic output
    config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class ConfigManager:
    """High-level config management."""

    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "config"
        self.agents_dir = self.base_dir / "agents"
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_global(self, name: str) -> Dict[str, Any]:
        """Load global config by name (ppo, neural, webots, training)."""
        if name in self._cache:
            return self._cache[name]

        path = self.config_dir / f"{name}_config.yaml"
        config = load_yaml(path)
        self._cache[name] = config
        return config

    def load_agent(self, agent_name: str) -> Dict[str, Any]:
        """Load complete agent config with inheritance."""
        agent_path = self.agents_dir / agent_name / "config.yaml"
        return load_config_with_inheritance(agent_path)

    def validate_agent(self, agent_name: str) -> Dict[str, List[str]]:
        """Validate agent config."""
        config = self.load_agent(agent_name)
        return validate_all_configs(config)

    def create_agent_config(
        self,
        agent_name: str,
        template: str = "default",
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create new agent config from template."""
        template_path = self.agents_dir / "template" / "config.yaml"
        template_config = load_yaml(template_path)

        # Apply overrides
        if overrides:
            template_config = deep_merge(template_config, overrides)

        # Set agent metadata
        template_config.setdefault("agent", {})
        template_config["agent"]["name"] = agent_name
        template_config["agent"]["created_at"] = __import__('datetime').datetime.now().isoformat()

        return template_config

    def save_agent_config(self, agent_name: str, config: Dict[str, Any]) -> Path:
        """Save agent config to file."""
        agent_dir = self.agents_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)

        config_path = agent_dir / "config.yaml"
        save_yaml(config, config_path)

        return config_path


# CLI Validation Entry Point
def main():
    """CLI entry point for config validation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.utils.config <config_path> [schema_name]")
        sys.exit(1)

    config_path = sys.argv[1]
    schema_name = sys.argv[2] if len(sys.argv) > 2 else "agent"

    config = load_yaml(config_path)
    errors = validate_config(config, schema_name)

    if errors:
        print(f"Validation FAILED for {schema_name}:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print(f"Validation PASSED for {schema_name}")


if __name__ == "__main__":
    main()