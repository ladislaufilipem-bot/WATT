# Curriculum Learning
from __future__ import annotations

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class CurriculumStage:
    """Single curriculum stage."""
    name: str
    timesteps: int  # Minimum timesteps before advancing
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    advancement_metric: str = "eval/mean_reward"  # Metric to check for advancement
    advancement_threshold: float = 0.0  # Threshold for advancement
    min_episodes: int = 100  # Minimum episodes in this stage
    max_episodes: Optional[int] = None  # Maximum episodes (None = unlimited)


class CurriculumScheduler:
    """
    Manages curriculum learning progression.

    Advances through stages based on performance metrics.
    """

    def __init__(self, stages: List[CurriculumStage]):
        self.stages = stages
        self.current_stage_idx = 0
        self.stage_timesteps = 0
        self.stage_episodes = 0
        self.stage_best_metric = float('-inf')

        # History
        self.stage_history = []

    @property
    def current_stage(self) -> Optional[CurriculumStage]:
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_stage_idx >= len(self.stages)

    def get_config_overrides(self) -> Dict[str, Any]:
        """Get current stage config overrides."""
        if self.current_stage:
            return self.current_stage.config_overrides.copy()
        return {}

    def step(self, timesteps: int, episodes: int = 0, metrics: Dict[str, float] = None) -> bool:
        """
        Update curriculum state.

        Returns True if stage advanced.
        """
        self.stage_timesteps += timesteps
        self.stage_episodes += episodes

        # Check advancement
        if self._should_advance(metrics):
            return self._advance_stage()

        return False

    def _should_advance(self, metrics: Dict[str, float] = None) -> bool:
        """Check if should advance to next stage."""
        stage = self.current_stage
        if not stage:
            return False

        # Check minimum timesteps
        if self.stage_timesteps < stage.timesteps:
            return False

        # Check minimum episodes
        if self.stage_episodes < stage.min_episodes:
            return False

        # Check maximum episodes
        if stage.max_episodes and self.stage_episodes >= stage.max_episodes:
            return True

        # Check metric threshold
        if metrics and stage.advancement_metric in metrics:
            current_metric = metrics[stage.advancement_metric]

            # Track best metric in this stage
            if current_metric > self.stage_best_metric:
                self.stage_best_metric = current_metric

            # Advance if threshold met
            if current_metric >= stage.advancement_threshold:
                return True

        return False

    def _advance_stage(self) -> bool:
        """Advance to next stage."""
        # Record stage completion
        self.stage_history.append({
            "stage": self.current_stage_idx,
            "name": self.current_stage.name if self.current_stage else "unknown",
            "timesteps": self.stage_timesteps,
            "episodes": self.stage_episodes,
            "best_metric": self.stage_best_metric
        })

        self.current_stage_idx += 1
        self.stage_timesteps = 0
        self.stage_episodes = 0
        self.stage_best_metric = float('-inf')

        return True

    def reset_stage(self) -> None:
        """Reset current stage counters."""
        self.stage_timesteps = 0
        self.stage_episodes = 0
        self.stage_best_metric = float('-inf')

    def get_progress(self) -> Dict[str, Any]:
        """Get curriculum progress info."""
        return {
            "current_stage": self.current_stage_idx,
            "total_stages": len(self.stages),
            "stage_name": self.current_stage.name if self.current_stage else "completed",
            "stage_timesteps": self.stage_timesteps,
            "stage_episodes": self.stage_episodes,
            "stage_best_metric": self.stage_best_metric,
            "is_complete": self.is_complete,
            "history": self.stage_history
        }


class AdaptiveCurriculum(CurriculumScheduler):
    """
    Curriculum that adapts based on agent performance.

    Adjusts difficulty based on success rate, reward, etc.
    """

    def __init__(
        self,
        stages: List[CurriculumStage],
        success_threshold: float = 0.8,
        reward_threshold: float = 0.0,
        patience: int = 3
    ):
        super().__init__(stages)
        self.success_threshold = success_threshold
        self.reward_threshold = reward_threshold
        self.patience = patience

        self.stagnation_counter = 0
        self.last_best_metric = float('-inf')

    def _should_advance(self, metrics: Dict[str, float] = None) -> bool:
        """Adaptive advancement logic."""
        stage = self.current_stage
        if not stage:
            return False

        # Base checks
        if self.stage_timesteps < stage.timesteps:
            return False
        if self.stage_episodes < stage.min_episodes:
            return False
        if stage.max_episodes and self.stage_episodes >= stage.max_episodes:
            return True

        # Adaptive checks based on metrics
        if metrics:
            success_rate = metrics.get("eval/success_rate", 0)
            mean_reward = metrics.get("eval/mean_reward", 0)

            # Check if performing well enough
            if success_rate >= self.success_threshold and mean_reward >= self.reward_threshold:
                self.stagnation_counter = 0
                return True

            # Track stagnation
            current_metric = metrics.get(stage.advancement_metric, 0)
            if current_metric > self.last_best_metric:
                self.last_best_metric = current_metric
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            # Force advance if stagnating
            if self.stagnation_counter >= self.patience:
                return True

        return False


def create_curriculum_from_config(config: Dict[str, Any]) -> CurriculumScheduler:
    """Create curriculum scheduler from config."""
    curriculum_config = config.get("orchestration", {}).get("curriculum", {})

    if not curriculum_config.get("enabled", False):
        return None

    stages = []
    for stage_config in curriculum_config.get("stages", []):
        stage = CurriculumStage(
            name=stage_config.get("name", "stage"),
            timesteps=stage_config.get("timesteps", 1_000_000),
            config_overrides=stage_config.get("config_overrides", {}),
            advancement_metric=stage_config.get("advancement_metric", "eval/mean_reward"),
            advancement_threshold=stage_config.get("advancement_threshold", 0.0),
            min_episodes=stage_config.get("min_episodes", 100),
            max_episodes=stage_config.get("max_episodes")
        )
        stages.append(stage)

    # Check if adaptive
    if curriculum_config.get("adaptive", False):
        return AdaptiveCurriculum(
            stages=stages,
            success_threshold=curriculum_config.get("success_threshold", 0.8),
            reward_threshold=curriculum_config.get("reward_threshold", 0.0),
            patience=curriculum_config.get("patience", 3)
        )

    return CurriculumScheduler(stages=stages)