"""
Rollout Integration Module

Provides integration between trajectory recording and the training loop.

This module wires together:
- Trajectory recording during agent execution
- Terminal verification after completion
- Export for training consumption
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable
from datetime import datetime

from .workflow import (
    WorkflowTrajectory,
    WorkflowStep,
    WorkflowMetadata,
    ToolObservation,
    TerminalOutcome,
    WorkflowRecorder,
)
from .verifiers import TerminalVerifier, WorkflowCompletionVerifier
from .terminal_rewards import (
    TerminalRewardComputer,
    TerminalRewardConfig,
    TerminalRewardResult,
)
from .export import (
    TrajectoryExporter,
    compute_dataset_stats,
    TrajectoryDatasetStats,
)


class RolloutConfig:
    """
    Configuration for rollout/trajectory collection.
    """
    
    def __init__(
        self,
        output_dir: str | Path = "./trajectories",
        auto_export: bool = True,
        export_format: str = "jsonl",
        verifier: TerminalVerifier | None = None,
        reward_config: TerminalRewardConfig | None = None,
        max_steps: int = 50,
        enable_logging: bool = True,
    ):
        """
        Args:
            output_dir: Directory for trajectory exports
            auto_export: Whether to automatically export after each trajectory
            export_format: Export format ("jsonl" or "json")
            verifier: Terminal verifier to use (defaults to WorkflowCompletionVerifier)
            reward_config: Terminal reward configuration
            max_steps: Maximum steps per trajectory
            enable_logging: Whether to print progress logs
        """
        self.output_dir = Path(output_dir)
        self.auto_export = auto_export
        self.export_format = export_format
        self.verifier = verifier or WorkflowCompletionVerifier(max_steps=max_steps)
        self.reward_config = reward_config or TerminalRewardConfig.default()
        self.max_steps = max_steps
        self.enable_logging = enable_logging


class RolloutManager:
    """
    Manages trajectory collection during rollouts/agent execution.
    
    This is the main integration point for wiring trajectory recording
    into your training loop or agent execution.
    
    Example:
        config = RolloutConfig(output_dir="./data/trajectories")
        manager = RolloutManager(config)
        
        # During rollout
        with manager.start_rollout("Fix the bug in main.py") as rollout:
            # Agent executes tools
            rollout.record_tool_call("read_file", {"path": "main.py"}, result="...")
            rollout.record_tool_call("edit_file", {...}, result="...")
            rollout.set_final_answer("Fixed the null pointer exception")
        
        # After rollout, trajectory is verified and exported
        print(f"Score: {manager.last_result.final_score}")
        
        # Get stats
        print(manager.get_stats())
    """
    
    def __init__(self, config: RolloutConfig | None = None):
        """
        Args:
            config: Rollout configuration
        """
        self.config = config or RolloutConfig()
        self.trajectories: list[WorkflowTrajectory] = []
        self.results: list[TerminalRewardResult] = []
        
        # Initialize components
        self.reward_computer = TerminalRewardComputer(
            verifier=self.config.verifier,
            config=self.config.reward_config,
        )
        self.exporter = TrajectoryExporter()
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current rollout state
        self._current_rollout: RolloutContext | None = None
    
    def start_rollout(
        self,
        user_request: str,
        system_prompt: str = "",
        tags: dict[str, str] | None = None,
    ) -> RolloutContext:
        """
        Start a new rollout/trajectory.
        
        Args:
            user_request: The user's request
            system_prompt: System prompt for the agent
            tags: Optional tags for filtering
        
        Returns:
            RolloutContext for recording the trajectory
        """
        trajectory = WorkflowTrajectory(
            user_request=user_request,
            system_prompt=system_prompt,
            tags=tags or {},
        )
        
        context = RolloutContext(trajectory, self)
        self._current_rollout = context
        
        if self.config.enable_logging:
            print(f"[Rollout] Started: {user_request[:50]}...")
        
        return context
    
    def _finish_rollout(self, trajectory: WorkflowTrajectory) -> TerminalRewardResult:
        """
        Called when a rollout context exits.
        
        Verifies the trajectory and computes terminal reward.
        """
        # Mark trajectory as complete
        if trajectory.ended_at is None:
            trajectory.ended_at = datetime.utcnow()
        
        # Compute terminal reward
        result = self.reward_computer.compute_and_attach(trajectory)
        
        # Store trajectory and result
        self.trajectories.append(trajectory)
        self.results.append(result)
        
        if self.config.enable_logging:
            status = "✓" if result.success else "✗"
            print(f"[Rollout] {status} Score: {result.final_score:.3f} - {result.explanation[:60]}")
        
        # Auto-export if enabled
        if self.config.auto_export:
            self._export_trajectory(trajectory)
        
        self._current_rollout = None
        return result
    
    def _export_trajectory(self, trajectory: WorkflowTrajectory) -> None:
        """Export a single trajectory."""
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        filename = f"trajectories_{timestamp}.{self.config.export_format}"
        filepath = self.config.output_dir / filename
        
        self.exporter.export_jsonl([trajectory], filepath, append=True)
    
    def export_all(self, filename: str | None = None) -> Path:
        """
        Export all collected trajectories.
        
        Args:
            filename: Optional custom filename
        
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectories_all_{timestamp}.{self.config.export_format}"
        
        filepath = self.config.output_dir / filename
        
        if self.config.export_format == "jsonl":
            self.exporter.export_jsonl(self.trajectories, filepath)
        else:
            self.exporter.export_json(self.trajectories, filepath)
        
        if self.config.enable_logging:
            print(f"[Rollout] Exported {len(self.trajectories)} trajectories to {filepath}")
        
        return filepath
    
    def get_stats(self) -> TrajectoryDatasetStats:
        """Get statistics over collected trajectories."""
        return compute_dataset_stats(self.trajectories)
    
    @property
    def last_trajectory(self) -> WorkflowTrajectory | None:
        """The most recently completed trajectory."""
        return self.trajectories[-1] if self.trajectories else None
    
    @property
    def last_result(self) -> TerminalRewardResult | None:
        """The most recent terminal reward result."""
        return self.results[-1] if self.results else None
    
    @property
    def success_rate(self) -> float:
        """Current success rate."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)
    
    @property
    def average_score(self) -> float:
        """Average terminal score."""
        if not self.results:
            return 0.0
        return sum(r.final_score for r in self.results) / len(self.results)


class RolloutContext:
    """
    Context manager for recording a single rollout/trajectory.
    
    Provides a convenient API for recording tool calls during execution.
    """
    
    def __init__(self, trajectory: WorkflowTrajectory, manager: RolloutManager):
        self.trajectory = trajectory
        self.manager = manager
        self._step_count = 0
    
    def __enter__(self) -> RolloutContext:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager._finish_rollout(self.trajectory)
        return False
    
    def record_tool_call(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
        result: Any = None,
        stdout: str | None = None,
        stderr: str | None = None,
        error: str | None = None,
        exit_code: int | None = None,
        reasoning: str | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
    ) -> WorkflowStep:
        """
        Record a tool call step.
        
        Args:
            tool_name: Name of the tool
            tool_arguments: Arguments passed to the tool
            result: Structured result from the tool
            stdout: Standard output
            stderr: Standard error
            error: Error message if tool failed
            exit_code: Exit code if applicable
            reasoning: Model's reasoning for this call
            started_at: When execution started
            ended_at: When execution ended
        
        Returns:
            The recorded WorkflowStep
        """
        observation = ToolObservation(
            result=result,
            stdout=stdout,
            stderr=stderr,
            error=error,
            exit_code=exit_code,
        )
        
        step = self.trajectory.add_step(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            observation=observation,
            reasoning=reasoning,
            started_at=started_at or datetime.utcnow(),
            ended_at=ended_at or datetime.utcnow(),
        )
        
        self._step_count += 1
        
        # Check max steps
        if self._step_count >= self.manager.config.max_steps:
            if self.manager.config.enable_logging:
                print(f"[Rollout] Warning: Reached max steps ({self.manager.config.max_steps})")
        
        return step
    
    def set_final_answer(self, answer: str) -> None:
        """Set the final answer for this trajectory."""
        self.trajectory.final_answer = answer
    
    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the trajectory."""
        self.trajectory.tags[key] = value
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set custom metadata."""
        self.trajectory.metadata.extra[key] = value


# Convenience function for quick setup
def create_rollout_manager(
    output_dir: str = "./trajectories",
    verifier: TerminalVerifier | None = None,
    auto_export: bool = True,
) -> RolloutManager:
    """
    Create a rollout manager with sensible defaults.
    
    Args:
        output_dir: Directory for trajectory exports
        verifier: Optional custom verifier
        auto_export: Whether to auto-export trajectories
    
    Returns:
        Configured RolloutManager
    """
    config = RolloutConfig(
        output_dir=output_dir,
        verifier=verifier,
        auto_export=auto_export,
    )
    return RolloutManager(config)
