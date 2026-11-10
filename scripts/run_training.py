#!/usr/bin/env python3
# Run Training Pipeline - Single or Multi-Agent

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utils.config import load_yaml
from core.utils.logging import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO training pipeline")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", help="Training config")
    parser.add_argument("--agents", type=str, help="Comma-separated agent names (default: all)")
    parser.add_argument("--parallel", type=int, default=1, help="Max parallel agents")
    parser.add_argument("--gpu", type=str, help="GPU IDs (e.g., '0,1')")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoints")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")
    return parser.parse_args()


def get_agent_dirs(agents_arg: str) -> list:
    """Get list of agent directories to train."""
    agents_root = PROJECT_ROOT / "agents"

    if agents_arg:
        agent_names = [a.strip() for a in agents_arg.split(",")]
        return [agents_root / name for name in agent_names if (agents_root / name).exists()]
    else:
        # Return all agent directories except template
        return [
            d for d in agents_root.iterdir()
            if d.is_dir() and d.name != "template" and (d / "config.yaml").exists()
        ]


def validate_agent_config(agent_dir: Path) -> bool:
    """Validate agent config."""
    config_path = agent_dir / "config.yaml"
    if not config_path.exists():
        return False

    try:
        config = load_yaml(config_path)
        required = ["agent", "ppo", "neural", "env", "training"]
        return all(key in config for key in required)
    except Exception:
        return False


def run_single_agent(agent_dir: Path, gpu_id: int = None, resume: bool = False, wandb: bool = True) -> subprocess.Popen:
    """Launch single agent training as subprocess."""
    train_script = agent_dir / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")

    cmd = [
        sys.executable, str(train_script),
        "--config", "config.yaml"
    ]

    if resume:
        cmd.append("--resume")
        # Find latest checkpoint
        ckpt_dir = agent_dir / "checkpoints"
        if ckpt_dir.exists():
            latest = ckpt_dir / "latest.pt"
            if latest.exists():
                cmd.extend(["--resume", str(latest)])

    if not wandb:
        cmd.append("--no-wandb")

    # Set GPU
    env = dict(subprocess.os.environ)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Run from agent directory
    proc = subprocess.Popen(
        cmd,
        cwd=str(agent_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    return proc


def monitor_processes(processes: dict, logger) -> dict:
    """Monitor running processes and return results."""
    results = {}

    while processes:
        for name, proc in list(processes.items()):
            # Read output
            line = proc.stdout.readline()
            if line:
                print(f"[{name}] {line.rstrip()}")

            # Check if done
            if proc.poll() is not None:
                returncode = proc.returncode
                results[name] = returncode
                del processes[name]

                if returncode == 0:
                    logger.info(f"{name} completed successfully")
                else:
                    logger.error(f"{name} failed with code {returncode}")

        if processes:
            import time
            time.sleep(1)

    return results


def main():
    args = parse_args()

    logger = get_logger(__name__)

    # Load training config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    training_config = load_yaml(config_path)

    # Get agents to train
    agent_dirs = get_agent_dirs(args.agents)

    if not agent_dirs:
        logger.error("No valid agents found to train")
        sys.exit(1)

    logger.info(f"Found {len(agent_dirs)} agents to train")
    for d in agent_dirs:
        valid = validate_agent_config(d)
        logger.info(f"  {d.name}: {'OK' if valid else 'INVALID CONFIG'}")

    # Filter valid agents
    agent_dirs = [d for d in agent_dirs if validate_agent_config(d)]

    if not agent_dirs:
        logger.error("No agents with valid configs")
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry run - config validation passed")
        return

    # Parse GPU IDs
    gpu_ids = None
    if args.gpu:
        gpu_ids = [int(x) for x in args.gpu.split(",")]

    # Launch training
    logger.info(f"Starting training for {len(agent_dirs)} agents (parallel={args.parallel})")

    running = {}
    completed = {}
    agent_queue = list(agent_dirs)

    # Start initial batch
    for i in range(min(args.parallel, len(agent_queue))):
        agent_dir = agent_queue.pop(0)
        gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else None

        logger.info(f"Starting {agent_dir.name} on GPU {gpu_id}")
        proc = run_single_agent(agent_dir, gpu_id=gpu_id, resume=args.resume, wandb=not args.no_wandb)
        running[agent_dir.name] = proc

    # Monitor and launch remaining
    while running or agent_queue:
        # Check for completed
        for name, proc in list(running.items()):
            if proc.poll() is not None:
                returncode = proc.returncode
                completed[name] = returncode
                del running[name]

                if returncode == 0:
                    logger.info(f"{name} completed successfully")
                else:
                    logger.error(f"{name} failed with code {returncode}")

        # Launch next if slots available
        while len(running) < args.parallel and agent_queue:
            agent_dir = agent_queue.pop(0)
            # Assign GPU round-robin
            gpu_id = None
            if gpu_ids:
                gpu_id = gpu_ids[len(running) % len(gpu_ids)]

            logger.info(f"Starting {agent_dir.name} on GPU {gpu_id}")
            proc = run_single_agent(agent_dir, gpu_id=gpu_id, resume=args.resume, wandb=not args.no_wandb)
            running[agent_dir.name] = proc

        if running:
            import time
            time.sleep(2)

    # Summary
    logger.info("=== Training Summary ===")
    for name, code in completed.items():
        status = "SUCCESS" if code == 0 else "FAILED"
        logger.info(f"  {name}: {status}")

    failed = [n for n, c in completed.items() if c != 0]
    if failed:
        logger.error(f"Failed agents: {failed}")
        sys.exit(1)
    else:
        logger.info("All agents completed successfully")


if __name__ == "__main__":
    main()