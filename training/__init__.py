# Training Package
"""
Training orchestration for multi-agent PPO
"""

from training.orchestrator import TrainingOrchestrator, AgentTrainingState, create_orchestrator
from training.evaluator import AgentEvaluator, EvaluationConfig, compare_agents, generate_evaluation_report
from training.curriculum import CurriculumScheduler, CurriculumStage, AdaptiveCurriculum, create_curriculum_from_config

__all__ = [
    "TrainingOrchestrator", "AgentTrainingState", "create_orchestrator",
    "AgentEvaluator", "EvaluationConfig", "compare_agents", "generate_evaluation_report",
    "CurriculumScheduler", "CurriculumStage", "AdaptiveCurriculum", "create_curriculum_from_config",
]