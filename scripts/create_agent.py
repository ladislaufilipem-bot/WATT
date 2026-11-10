#!/usr/bin/env python3
# Create New Agent from Template

import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utils.config import load_yaml, save_yaml, deep_merge


TEMPLATES = {
    "walker": {
        "description": "Bipedal/quadruped walker",
        "observation_keys": ["joint_pos", "joint_vel", "imu", "lidar"],
        "action_keys": ["motor_pos"],
        "reward_weights": {
            "forward_vel": 1.0,
            "energy": -0.01,
            "alive": 0.1,
            "stability": 0.5
        },
        "observation_space": {
            "joint_pos": {"shape": [12], "dtype": "float32", "min": -3.14, "max": 3.14},
            "joint_vel": {"shape": [12], "dtype": "float32", "min": -10.0, "max": 10.0},
            "imu": {"shape": [3], "dtype": "float32", "min": -3.14, "max": 3.14},
            "lidar": {"shape": [360], "dtype": "float32", "min": 0.0, "max": 10.0}
        },
        "action_space": {
            "motor_pos": {"shape": [12], "dtype": "float32", "min": -3.14, "max": 3.14}
        }
    },
    "arm": {
        "description": "Robotic arm manipulation",
        "observation_keys": ["joint_pos", "joint_vel", "ee_pos", "ee_quat", "target_pos"],
        "action_keys": ["joint_pos"],
        "reward_weights": {
            "target_distance": -1.0,
            "energy": -0.001,
            "alive": 0.05,
            "success": 10.0
        },
        "observation_space": {
            "joint_pos": {"shape": [6], "dtype": "float32", "min": -3.14, "max": 3.14},
            "joint_vel": {"shape": [6], "dtype": "float32", "min": -5.0, "max": 5.0},
            "ee_pos": {"shape": [3], "dtype": "float32", "min": -1.0, "max": 1.0},
            "ee_quat": {"shape": [4], "dtype": "float32", "min": -1.0, "max": 1.0},
            "target_pos": {"shape": [3], "dtype": "float32", "min": -1.0, "max": 1.0}
        },
        "action_space": {
            "joint_pos": {"shape": [6], "dtype": "float32", "min": -3.14, "max": 3.14}
        }
    },
    "drone": {
        "description": "Quadcopter drone",
        "observation_keys": ["pos", "vel", "quat", "ang_vel", "lidar"],
        "action_keys": ["motor_rpm"],
        "reward_weights": {
            "target_distance": -1.0,
            "stability": 0.5,
            "energy": -0.001,
            "alive": 0.1
        },
        "observation_space": {
            "pos": {"shape": [3], "dtype": "float32", "min": -10.0, "max": 10.0},
            "vel": {"shape": [3], "dtype": "float32", "min": -5.0, "max": 5.0},
            "quat": {"shape": [4], "dtype": "float32", "min": -1.0, "max": 1.0},
            "ang_vel": {"shape": [3], "dtype": "float32", "min": -10.0, "max": 10.0},
            "lidar": {"shape": [360], "dtype": "float32", "min": 0.0, "max": 20.0}
        },
        "action_space": {
            "motor_rpm": {"shape": [4], "dtype": "float32", "min": 0.0, "max": 2000.0}
        }
    },
    "default": {
        "description": "Generic agent",
        "observation_keys": ["joint_pos", "joint_vel", "imu"],
        "action_keys": ["motor_pos"],
        "reward_weights": {
            "forward_vel": 1.0,
            "energy": -0.01,
            "alive": 0.1
        },
        "observation_space": {
            "joint_pos": {"shape": [12], "dtype": "float32", "min": -3.14, "max": 3.14},
            "joint_vel": {"shape": [12], "dtype": "float32", "min": -10.0, "max": 10.0},
            "imu": {"shape": [3], "dtype": "float32", "min": -3.14, "max": 3.14}
        },
        "action_space": {
            "motor_pos": {"shape": [12], "dtype": "float32", "min": -3.14, "max": 3.14}
        }
    }
}


def list_templates():
    """Print available templates."""
    print("Available templates:")
    for name, info in TEMPLATES.items():
        print(f"  {name:10s} - {info['description']}")


def create_agent(
    name: str,
    template: str = "default",
    description: str = "",
    world: str = None,
    output_dir: Path = None
) -> Path:
    """Create new agent from template."""

    if template not in TEMPLATES:
        raise ValueError(f"Unknown template: {template}. Available: {list(TEMPLATES.keys())}")

    template_info = TEMPLATES[template]

    # Determine output directory
    if output_dir is None:
        output_dir = PROJECT_ROOT / "agents" / name

    if output_dir.exists():
        raise FileExistsError(f"Agent directory already exists: {output_dir}")

    # Copy template directory
    template_dir = PROJECT_ROOT / "agents" / "template"
    shutil.copytree(template_dir, output_dir)

    # Load template config
    config_path = output_dir / "config.yaml"
    config = load_yaml(config_path)

    # Update agent metadata
    config["agent"]["name"] = name
    config["agent"]["description"] = description or template_info["description"]
    config["agent"]["created_at"] = datetime.now().isoformat()
    config["agent"]["template"] = template

    # Apply template-specific config
    config["env"]["observation_keys"] = template_info["observation_keys"]
    config["env"]["action_keys"] = template_info["action_keys"]
    config["env"]["reward_weights"] = template_info["reward_weights"]
    config["env"]["observation_space"] = template_info["observation_space"]
    config["env"]["action_space"] = template_info["action_space"]

    if world:
        config["env"]["world"] = world

    # Save updated config
    save_yaml(config, config_path)

    # Create subdirectories
    (output_dir / "model").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)

    # Create model __init__.py
    model_init = output_dir / "model" / "__init__.py"
    model_init.write_text(f'''# Custom model for {name}
# Override this file to customize network architecture

from core.neural.actor import create_actor
from core.neural.critic import create_critic
from core.neural.encoders import create_encoder


def create_custom_actor(config, obs_dim, action_dim):
    """Create custom actor - override for custom architectures."""
    return create_actor(config, obs_dim, action_dim)


def create_custom_critic(config, obs_dim, action_dim):
    """Create custom critic - override for custom architectures."""
    return create_critic(config, obs_dim, action_dim)
''')

    # Update train.py with agent-specific info
    train_py = output_dir / "train.py"
    content = train_py.read_text()
    content = content.replace(
        'AGENT_NAME = "template"',
        f'AGENT_NAME = "{name}"'
    )
    content = content.replace(
        'from core.ppo.agent import create_agent_from_config',
        f'from core.ppo.agent import create_agent_from_config\nAGENT_NAME = "{name}"'
    )
    train_py.write_text(content)

    print(f"Created agent: {name}")
    print(f"  Directory: {output_dir}")
    print(f"  Template: {template} ({template_info['description']})")
    print(f"  Config: {config_path}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Create new PPO agent from template")
    parser.add_argument("--name", "-n", required=True, help="Agent name (e.g., agent_001)")
    parser.add_argument("--template", "-t", default="default", help="Template to use")
    parser.add_argument("--description", "-d", default="", help="Agent description")
    parser.add_argument("--world", "-w", help="Webots world file")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: agents/<name>)")
    parser.add_argument("--list", "-l", action="store_true", help="List available templates")

    args = parser.parse_args()

    if args.list:
        list_templates()
        return

    # Validate name format
    if not args.name.startswith("agent_"):
        print("Warning: Agent name should follow format 'agent_XXX' (e.g., agent_001)")

    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        create_agent(
            name=args.name,
            template=args.template,
            description=args.description,
            world=args.world,
            output_dir=output_dir
        )
        print("\nNext steps:")
        print(f"  1. Customize reward function in {output_dir}/train.py")
        print(f"  2. Adjust config in {output_dir}/config.yaml")
        print(f"  3. Run training: python {output_dir}/train.py")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()