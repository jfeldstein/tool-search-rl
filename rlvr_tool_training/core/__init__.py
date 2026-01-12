"""Core module for RLVR training."""

# Original single-step trajectory support
from .trajectories import Trajectory, ToolCall, VerifiableReward, TrajectoryGenerator
from .rewards import RewardFunction, ToolSelectionReward, ParameterAccuracyReward
from .trainer import RLVRTrainer

# New: Multi-step workflow trajectory support
from .workflow import (
    WorkflowTrajectory,
    WorkflowStep,
    WorkflowMetadata,
    ToolObservation,
    TerminalOutcome,
    WorkflowRecorder,
)

# New: Terminal verifiers
from .verifiers import (
    TerminalVerifier,
    PytestVerifier,
    ArtifactVerifier,
    FunctionVerifier,
    CompositeVerifier,
    WorkflowCompletionVerifier,
)

# New: Terminal reward computation
from .terminal_rewards import (
    TerminalRewardComputer,
    TerminalRewardConfig,
    TerminalRewardResult,
    ShapingPenalty,
    compute_terminal_reward,
    too_many_calls_penalty,
    excessive_latency_penalty,
    tool_error_penalty,
)

# New: Export and statistics
from .export import (
    TrajectoryExporter,
    TrajectoryLoader,
    TrajectoryDatasetStats,
    compute_dataset_stats,
    export_trajectories_jsonl,
    load_trajectories_jsonl,
)

# New: Rollout integration
from .rollout import (
    RolloutManager,
    RolloutConfig,
    RolloutContext,
    create_rollout_manager,
)

__all__ = [
    # Original
    "Trajectory",
    "ToolCall", 
    "VerifiableReward",
    "TrajectoryGenerator",
    "RewardFunction",
    "ToolSelectionReward",
    "ParameterAccuracyReward",
    "RLVRTrainer",
    # Workflow trajectories
    "WorkflowTrajectory",
    "WorkflowStep",
    "WorkflowMetadata",
    "ToolObservation",
    "TerminalOutcome",
    "WorkflowRecorder",
    # Verifiers
    "TerminalVerifier",
    "PytestVerifier",
    "ArtifactVerifier",
    "FunctionVerifier",
    "CompositeVerifier",
    "WorkflowCompletionVerifier",
    # Terminal rewards
    "TerminalRewardComputer",
    "TerminalRewardConfig",
    "TerminalRewardResult",
    "ShapingPenalty",
    "compute_terminal_reward",
    "too_many_calls_penalty",
    "excessive_latency_penalty",
    "tool_error_penalty",
    # Export
    "TrajectoryExporter",
    "TrajectoryLoader",
    "TrajectoryDatasetStats",
    "compute_dataset_stats",
    "export_trajectories_jsonl",
    "load_trajectories_jsonl",
    # Rollout
    "RolloutManager",
    "RolloutConfig",
    "RolloutContext",
    "create_rollout_manager",
]
