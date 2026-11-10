#!/usr/bin/env python3
# Visualize Training Logs and Compare Agents

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from core.utils.logging import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument("--agents", type=str, help="Comma-separated agent names")
    parser.add_argument("--compare", type=str, help="Comma-separated agents to compare")
    parser.add_argument("--metric", type=str, default="episode_reward", help="Metric to plot")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--output", type=str, help="Output file (HTML or PNG)")
    parser.add_argument("--live", action="store_true", help="Live updating plot")
    parser.add_argument("--window", type=int, default=100, help="Rolling window size")
    parser.add_argument("--format", type=str, default="png", choices=["png", "html", "pdf"], help="Output format")
    return parser.parse_args()


def load_jsonl_logs(log_path: Path) -> list:
    """Load JSONL log file."""
    logs = []
    if not log_path.exists():
        return logs

    with open(log_path, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return logs


def find_agent_logs(agent_name: str, log_dir: Path) -> list:
    """Find all log files for an agent."""
    logs = []
    pattern = f"train_*{agent_name}*.jsonl"
    for log_file in log_dir.glob(pattern):
        logs.extend(load_jsonl_logs(log_file))
    return sorted(logs, key=lambda x: x.get("timestamp", ""))


def extract_metrics(logs: list, metric: str, window: int = 100) -> dict:
    """Extract metric time series from logs."""
    steps = []
    values = []
    episodes = []
    ep_rewards = []

    for log in logs:
        if "step" in log and log.get("metrics"):
            metrics = log["metrics"]
            if metric in metrics:
                steps.append(log["step"])
                values.append(metrics[metric])

        if "episode" in log and log.get("metrics"):
            metrics = log["metrics"]
            if "episode_reward" in metrics:
                episodes.append(log["episode"])
                ep_rewards.append(metrics["episode_reward"])

    # Compute rolling average
    rolling_mean = []
    rolling_std = []
    if len(values) >= window:
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_vals = values[start:i+1]
            rolling_mean.append(np.mean(window_vals))
            rolling_std.append(np.std(window_vals))
    else:
        rolling_mean = values
        rolling_std = [0] * len(values)

    return {
        "steps": steps,
        "values": values,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "episodes": episodes,
        "ep_rewards": ep_rewards
    }


def plot_agent_metrics(agent_name: str, metrics: dict, ax, metric: str, window: int):
    """Plot metrics for single agent."""
    steps = metrics["steps"]
    values = metrics["values"]
    rolling_mean = metrics["rolling_mean"]
    rolling_std = metrics["rolling_std"]

    if not steps:
        ax.text(0.5, 0.5, f"No data for {agent_name}", ha='center', va='center', transform=ax.transAxes)
        return

    # Raw values (light)
    ax.plot(steps, values, alpha=0.2, color='blue', linewidth=0.5)

    # Rolling mean
    ax.plot(steps, rolling_mean, color='blue', linewidth=1.5, label=f'{agent_name}')

    # Confidence band
    upper = np.array(rolling_mean) + np.array(rolling_std)
    lower = np.array(rolling_mean) - np.array(rolling_std)
    ax.fill_between(steps, lower, upper, alpha=0.2, color='blue')

    ax.set_xlabel('Steps')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{agent_name} - {metric}')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_comparison(agents_metrics: dict, metric: str, window: int, output_path: Path):
    """Create comparison plot for multiple agents."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not installed, skipping plot")
        return

    n_agents = len(agents_metrics)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Agent Comparison - {metric}', fontsize=16)

    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

    # Plot 1: Rolling mean comparison
    ax1 = axes[0, 0]
    for i, (name, metrics) in enumerate(agents_metrics.items()):
        steps = metrics["steps"]
        rolling_mean = metrics["rolling_mean"]
        rolling_std = metrics["rolling_std"]

        if steps:
            ax1.plot(steps, rolling_mean, color=colors[i], linewidth=1.5, label=name)
            upper = np.array(rolling_mean) + np.array(rolling_std)
            lower = np.array(rolling_mean) - np.array(rolling_std)
            ax1.fill_between(steps, lower, upper, alpha=0.2, color=colors[i])

    ax1.set_xlabel('Steps')
    ax1.set_ylabel(f'{metric} (rolling {window})')
    ax1.set_title('Rolling Mean Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Episode rewards
    ax2 = axes[0, 1]
    for i, (name, metrics) in enumerate(agents_metrics.items()):
        episodes = metrics["episodes"]
        ep_rewards = metrics["ep_rewards"]

        if episodes:
            ax2.plot(episodes, ep_rewards, 'o-', color=colors[i], linewidth=1, markersize=3, label=name, alpha=0.7)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Reward')
    ax2.set_title('Episode Rewards')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final performance distribution
    ax3 = axes[1, 0]
    agent_names = []
    final_means = []
    final_stds = []

    for name, metrics in agents_metrics.items():
        if metrics["values"]:
            # Last 10% of values
            n = max(1, len(metrics["values"]) // 10)
            recent = metrics["values"][-n:]
            agent_names.append(name)
            final_means.append(np.mean(recent))
            final_stds.append(np.std(recent))

    if agent_names:
        x_pos = np.arange(len(agent_names))
        ax3.bar(x_pos, final_means, yerr=final_stds, capsize=5, color=colors[:len(agent_names)], alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(agent_names, rotation=45, ha='right')
        ax3.set_ylabel(f'Final {metric} (last 10%)')
        ax3.set_title('Final Performance')
        ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Sample efficiency (reward at fixed steps)
    ax4 = axes[1, 1]
    # Find common step milestones
    all_steps = set()
    for metrics in agents_metrics.values():
        all_steps.update(metrics["steps"])

    if all_steps:
        milestones = sorted(all_steps)[::max(1, len(all_steps)//10)]

        for i, (name, metrics) in enumerate(agents_metrics.items()):
            steps = metrics["steps"]
            values = metrics["values"]

            if steps:
                # Interpolate at milestones
                interp_values = np.interp(milestones, steps, values)
                ax4.plot(milestones, interp_values, 'o-', color=colors[i], linewidth=1.5, label=name)

        ax4.set_xlabel('Steps')
        ax4.set_ylabel(metric.replace('_', ' ').title())
        ax4.set_title('Sample Efficiency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to {output_path}")


def plot_single_agent(agent_name: str, metrics: dict, metric: str, window: int, output_path: Path):
    """Create detailed plot for single agent."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not installed, skipping plot")
        return

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    steps = metrics["steps"]
    values = metrics["values"]
    rolling_mean = metrics["rolling_mean"]
    rolling_std = metrics["rolling_std"]
    episodes = metrics["episodes"]
    ep_rewards = metrics["ep_rewards"]

    # Main learning curve
    ax1 = fig.add_subplot(gs[0, :])
    if steps:
        ax1.plot(steps, values, alpha=0.2, color='blue', linewidth=0.5, label='Raw')
        ax1.plot(steps, rolling_mean, color='blue', linewidth=2, label=f'Rolling Mean ({window})')
        upper = np.array(rolling_mean) + np.array(rolling_std)
        lower = np.array(rolling_mean) - np.array(rolling_std)
        ax1.fill_between(steps, lower, upper, alpha=0.2, color='blue')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel(metric.replace('_', ' ').title())
    ax1.set_title(f'{agent_name} - Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Episode rewards
    ax2 = fig.add_subplot(gs[1, 0])
    if episodes:
        ax2.plot(episodes, ep_rewards, 'o-', color='green', linewidth=1, markersize=3, alpha=0.7)
        # Rolling average of episodes
        if len(ep_rewards) > 10:
            ep_rolling = np.convolve(ep_rewards, np.ones(10)/10, mode='valid')
            ax2.plot(episodes[9:], ep_rolling, color='darkgreen', linewidth=2, label='Rolling (10)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Reward')
    ax2.set_title('Episode Rewards')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Reward distribution
    ax3 = fig.add_subplot(gs[1, 1])
    if ep_rewards:
        ax3.hist(ep_rewards, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(np.mean(ep_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(ep_rewards):.2f}')
        ax3.axvline(np.median(ep_rewards), color='orange', linestyle='--', label=f'Median: {np.median(ep_rewards):.2f}')
    ax3.set_xlabel('Episode Reward')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Reward Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Step-wise statistics
    ax4 = fig.add_subplot(gs[2, 0])
    if steps and len(steps) > window:
        # Compute statistics in windows
        window_size = max(1, len(steps) // 20)
        win_means = []
        win_stds = []
        win_steps = []

        for i in range(0, len(steps), window_size):
            end = min(i + window_size, len(steps))
            win_steps.append(steps[i])
            win_means.append(np.mean(values[i:end]))
            win_stds.append(np.std(values[i:end]))

        ax4.errorbar(win_steps, win_means, yerr=win_stds, fmt='o-', capsize=3, color='red')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel(f'{metric} (windowed)')
    ax4.set_title('Windowed Statistics')
    ax4.grid(True, alpha=0.3)

    # Progress summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    if values:
        summary_text = f"""
        Summary Statistics:

        Total Steps: {steps[-1] if steps else 0:,}
        Total Episodes: {len(episodes)}

        {metric}:
        - Mean: {np.mean(values):.3f}
        - Std: {np.std(values):.3f}
        - Min: {np.min(values):.3f}
        - Max: {np.max(values):.3f}
        - Final (last 10%): {np.mean(values[-max(1,len(values)//10):]):.3f}

        Episode Reward:
        - Mean: {np.mean(ep_rewards):.3f}
        - Std: {np.std(ep_rewards):.3f}
        - Best: {np.max(ep_rewards):.3f}
        """
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{agent_name} - Training Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Single agent plot saved to {output_path}")


def generate_html_report(agents_metrics: dict, metric: str, output_path: Path):
    """Generate interactive HTML report using Plotly (if available)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rolling Mean Comparison', 'Episode Rewards',
                          'Final Performance', 'Sample Efficiency')
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, (name, metrics) in enumerate(agents_metrics.items()):
            color = colors[i % len(colors)]
            steps = metrics["steps"]
            rolling_mean = metrics["rolling_mean"]
            rolling_std = metrics["rolling_std"]
            episodes = metrics["episodes"]
            ep_rewards = metrics["ep_rewards"]

            if steps:
                # Rolling mean
                fig.add_trace(
                    go.Scatter(x=steps, y=rolling_mean, name=name,
                              line=dict(color=color), showlegend=True),
                    row=1, col=1
                )
                # Confidence band
                upper = np.array(rolling_mean) + np.array(rolling_std)
                lower = np.array(rolling_mean) - np.array(rolling_std)
                fig.add_trace(
                    go.Scatter(x=steps+steps[::-1], y=np.concatenate([upper, lower[::-1]]),
                              fill='toself', fillcolor=color, opacity=0.2,
                              line=dict(color='rgba(255,255,255,0)'),
                              name=f'{name} ±std', showlegend=False),
                    row=1, col=1
                )

            if episodes:
                fig.add_trace(
                    go.Scatter(x=episodes, y=ep_rewards, name=name,
                              mode='lines+markers', line=dict(color=color),
                              marker=dict(size=4), showlegend=False),
                    row=1, col=2
                )

        # Final performance bar chart
        agent_names = []
        final_means = []
        final_stds = []

        for name, metrics in agents_metrics.items():
            if metrics["values"]:
                n = max(1, len(metrics["values"]) // 10)
                recent = metrics["values"][-n:]
                agent_names.append(name)
                final_means.append(np.mean(recent))
                final_stds.append(np.std(recent))

        if agent_names:
            fig.add_trace(
                go.Bar(x=agent_names, y=final_means, error_y=dict(type='data', array=final_stds),
                      name='Final Performance', showlegend=False),
                row=2, col=1
            )

        fig.update_layout(height=800, title_text=f'Agent Comparison - {metric}')
        fig.write_html(str(output_path))
        print(f"HTML report saved to {output_path}")

    except ImportError:
        print("Plotly not installed, skipping HTML report")


def main():
    args = parse_args()

    logger = get_logger(__name__)
    log_dir = PROJECT_ROOT / args.log_dir

    if args.compare:
        agent_names = [a.strip() for a in args.compare.split(",")]
    elif args.agents:
        agent_names = [a.strip() for a in args.agents.split(",")]
    else:
        # Find all agents with logs
        agent_names = []
        for log_file in log_dir.glob("train_*.jsonl"):
            # Extract agent name from filename
            name = log_file.stem.replace("train_", "").split("_")[0]
            if name not in agent_names:
                agent_names.append(name)

    if not agent_names:
        logger.error("No agents found with logs")
        sys.exit(1)

    logger.info(f"Loading logs for agents: {agent_names}")

    # Load metrics for each agent
    agents_metrics = {}
    for name in agent_names:
        logs = find_agent_logs(name, log_dir)
        metrics = extract_metrics(logs, args.metric, args.window)
        agents_metrics[name] = metrics
        logger.info(f"  {name}: {len(metrics['steps'])} steps, {len(metrics['episodes'])} episodes")

    # Generate output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "reports"
        output_dir.mkdir(exist_ok=True)

        if len(agent_names) == 1:
            output_path = output_dir / f"{agent_names[0]}_{args.metric}_{timestamp}.{args.format}"
        else:
            output_path = output_dir / f"comparison_{args.metric}_{timestamp}.{args.format}"

    if args.format == "html":
        generate_html_report(agents_metrics, args.metric, output_path)
    elif args.compare or len(agent_names) > 1:
        plot_comparison(agents_metrics, args.metric, args.window, output_path)
    else:
        plot_single_agent(agent_names[0], agents_metrics[agent_names[0]], args.metric, args.window, output_path)

    logger.info(f"Visualization complete: {output_path}")


if __name__ == "__main__":
    main()