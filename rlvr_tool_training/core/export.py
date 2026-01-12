"""
Export and Statistics Module

Provides utilities for:
- Exporting trajectories to JSONL format
- Computing statistics over trajectory datasets
- Loading trajectories from files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator
from dataclasses import dataclass, field
from datetime import datetime

from .workflow import WorkflowTrajectory, TerminalOutcome


@dataclass
class TrajectoryDatasetStats:
    """
    Statistics computed over a dataset of trajectories.
    """
    # Counts
    total_trajectories: int = 0
    completed_trajectories: int = 0
    successful_trajectories: int = 0
    failed_trajectories: int = 0
    
    # Success rate
    success_rate: float = 0.0
    
    # Score statistics
    score_mean: float = 0.0
    score_std: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0
    
    # Step statistics
    avg_steps: float = 0.0
    min_steps: int = 0
    max_steps: int = 0
    total_steps: int = 0
    
    # Latency statistics (ms)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Error statistics
    total_tool_errors: int = 0
    trajectories_with_errors: int = 0
    
    # Tool usage distribution
    tool_usage: dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "counts": {
                "total": self.total_trajectories,
                "completed": self.completed_trajectories,
                "successful": self.successful_trajectories,
                "failed": self.failed_trajectories,
            },
            "success_rate": self.success_rate,
            "scores": {
                "mean": self.score_mean,
                "std": self.score_std,
                "min": self.score_min,
                "max": self.score_max,
            },
            "steps": {
                "avg": self.avg_steps,
                "min": self.min_steps,
                "max": self.max_steps,
                "total": self.total_steps,
            },
            "latency_ms": {
                "avg": self.avg_latency_ms,
                "min": self.min_latency_ms,
                "max": self.max_latency_ms,
                "total": self.total_latency_ms,
            },
            "errors": {
                "total_tool_errors": self.total_tool_errors,
                "trajectories_with_errors": self.trajectories_with_errors,
            },
            "tool_usage": self.tool_usage,
        }
    
    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "Trajectory Dataset Statistics",
            "=" * 40,
            f"Total trajectories: {self.total_trajectories}",
            f"Completed: {self.completed_trajectories}",
            f"Success rate: {self.success_rate:.1%}",
            "",
            f"Score: mean={self.score_mean:.3f}, std={self.score_std:.3f}, range=[{self.score_min:.3f}, {self.score_max:.3f}]",
            f"Steps: avg={self.avg_steps:.1f}, range=[{self.min_steps}, {self.max_steps}]",
            f"Latency: avg={self.avg_latency_ms:.0f}ms, total={self.total_latency_ms:.0f}ms",
            "",
            f"Tool errors: {self.total_tool_errors} total in {self.trajectories_with_errors} trajectories",
        ]
        
        if self.tool_usage:
            lines.append("")
            lines.append("Tool usage:")
            for tool, count in sorted(self.tool_usage.items(), key=lambda x: -x[1]):
                lines.append(f"  {tool}: {count}")
        
        return "\n".join(lines)


def compute_dataset_stats(trajectories: list[WorkflowTrajectory]) -> TrajectoryDatasetStats:
    """
    Compute statistics over a list of trajectories.
    
    Args:
        trajectories: List of workflow trajectories
    
    Returns:
        TrajectoryDatasetStats with computed statistics
    """
    stats = TrajectoryDatasetStats()
    
    if not trajectories:
        return stats
    
    stats.total_trajectories = len(trajectories)
    
    scores = []
    steps_list = []
    latencies = []
    
    for traj in trajectories:
        # Count completed/successful
        if traj.is_complete:
            stats.completed_trajectories += 1
            if traj.succeeded:
                stats.successful_trajectories += 1
            else:
                stats.failed_trajectories += 1
            
            if traj.terminal_score is not None:
                scores.append(traj.terminal_score)
        
        # Step counts
        steps_list.append(traj.num_steps)
        stats.total_steps += traj.num_steps
        
        # Latency
        if traj.metadata.total_latency_ms > 0:
            latencies.append(traj.metadata.total_latency_ms)
            stats.total_latency_ms += traj.metadata.total_latency_ms
        
        # Errors
        if traj.metadata.tool_errors > 0:
            stats.total_tool_errors += traj.metadata.tool_errors
            stats.trajectories_with_errors += 1
        
        # Tool usage
        for step in traj.steps:
            stats.tool_usage[step.tool_name] = stats.tool_usage.get(step.tool_name, 0) + 1
    
    # Compute success rate
    if stats.completed_trajectories > 0:
        stats.success_rate = stats.successful_trajectories / stats.completed_trajectories
    
    # Compute score statistics
    if scores:
        stats.score_mean = sum(scores) / len(scores)
        stats.score_min = min(scores)
        stats.score_max = max(scores)
        if len(scores) > 1:
            variance = sum((s - stats.score_mean) ** 2 for s in scores) / (len(scores) - 1)
            stats.score_std = variance ** 0.5
    
    # Compute step statistics
    if steps_list:
        stats.avg_steps = sum(steps_list) / len(steps_list)
        stats.min_steps = min(steps_list)
        stats.max_steps = max(steps_list)
    
    # Compute latency statistics
    if latencies:
        stats.avg_latency_ms = sum(latencies) / len(latencies)
        stats.min_latency_ms = min(latencies)
        stats.max_latency_ms = max(latencies)
    
    return stats


class TrajectoryExporter:
    """
    Exports trajectories to various formats.
    
    Primary format is JSONL (one JSON object per line) which is
    easy to process and widely supported.
    
    Example:
        exporter = TrajectoryExporter()
        exporter.export_jsonl(trajectories, "trajectories.jsonl")
    """
    
    def __init__(self, include_metadata: bool = True):
        """
        Args:
            include_metadata: Whether to include full metadata in exports
        """
        self.include_metadata = include_metadata
    
    def export_jsonl(
        self,
        trajectories: list[WorkflowTrajectory],
        output_path: str | Path,
        append: bool = False,
    ) -> int:
        """
        Export trajectories to JSONL file.
        
        Each line contains:
        - Full trajectory trace
        - Terminal reward/score
        - Metadata
        
        Args:
            trajectories: Trajectories to export
            output_path: Output file path
            append: If True, append to existing file
        
        Returns:
            Number of trajectories exported
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = "a" if append else "w"
        count = 0
        
        with open(output_path, mode) as f:
            for traj in trajectories:
                record = self._trajectory_to_record(traj)
                f.write(json.dumps(record, default=str) + "\n")
                count += 1
        
        return count
    
    def export_json(
        self,
        trajectories: list[WorkflowTrajectory],
        output_path: str | Path,
    ) -> int:
        """
        Export trajectories to single JSON file (array format).
        
        Args:
            trajectories: Trajectories to export
            output_path: Output file path
        
        Returns:
            Number of trajectories exported
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        records = [self._trajectory_to_record(traj) for traj in trajectories]
        
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        
        return len(records)
    
    def _trajectory_to_record(self, traj: WorkflowTrajectory) -> dict[str, Any]:
        """Convert trajectory to export record."""
        record = {
            "trajectory_id": traj.trajectory_id,
            "user_request": traj.user_request,
            "system_prompt": traj.system_prompt,
            "final_answer": traj.final_answer,
            "num_steps": traj.num_steps,
            "started_at": traj.started_at.isoformat() if traj.started_at else None,
            "ended_at": traj.ended_at.isoformat() if traj.ended_at else None,
            "tags": traj.tags,
        }
        
        # Add steps
        record["steps"] = [
            {
                "step_index": step.step_index,
                "tool_name": step.tool_name,
                "tool_arguments": step.tool_arguments,
                "observation": {
                    "stdout": step.observation.stdout,
                    "stderr": step.observation.stderr,
                    "result": step.observation.result,
                    "error": step.observation.error,
                    "exit_code": step.observation.exit_code,
                },
                "reasoning": step.reasoning,
                "duration_ms": step.duration_ms,
            }
            for step in traj.steps
        ]
        
        # Add terminal outcome (THE key field for training)
        if traj.outcome:
            record["terminal_outcome"] = {
                "success": traj.outcome.success,
                "score": traj.outcome.score,
                "explanation": traj.outcome.explanation,
                "verifier_name": traj.outcome.verifier_name,
            }
        else:
            record["terminal_outcome"] = None
        
        # Add metadata
        if self.include_metadata:
            record["metadata"] = {
                "total_tool_calls": traj.metadata.total_tool_calls,
                "total_latency_ms": traj.metadata.total_latency_ms,
                "tool_errors": traj.metadata.tool_errors,
                "schema_violations": traj.metadata.schema_violations,
                "extra": traj.metadata.extra,
            }
        
        return record


class TrajectoryLoader:
    """
    Loads trajectories from files.
    
    Example:
        loader = TrajectoryLoader()
        trajectories = loader.load_jsonl("trajectories.jsonl")
    """
    
    def load_jsonl(self, input_path: str | Path) -> list[WorkflowTrajectory]:
        """
        Load trajectories from JSONL file.
        
        Args:
            input_path: Path to JSONL file
        
        Returns:
            List of WorkflowTrajectory objects
        """
        input_path = Path(input_path)
        trajectories = []
        
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    traj = self._record_to_trajectory(record)
                    trajectories.append(traj)
        
        return trajectories
    
    def load_json(self, input_path: str | Path) -> list[WorkflowTrajectory]:
        """
        Load trajectories from JSON file (array format).
        
        Args:
            input_path: Path to JSON file
        
        Returns:
            List of WorkflowTrajectory objects
        """
        input_path = Path(input_path)
        
        with open(input_path) as f:
            records = json.load(f)
        
        return [self._record_to_trajectory(r) for r in records]
    
    def iter_jsonl(self, input_path: str | Path) -> Iterator[WorkflowTrajectory]:
        """
        Iterate over trajectories from JSONL file (memory efficient).
        
        Args:
            input_path: Path to JSONL file
        
        Yields:
            WorkflowTrajectory objects
        """
        input_path = Path(input_path)
        
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    yield self._record_to_trajectory(record)
    
    def _record_to_trajectory(self, record: dict[str, Any]) -> WorkflowTrajectory:
        """Convert export record back to trajectory."""
        from .workflow import (
            WorkflowTrajectory,
            WorkflowStep,
            WorkflowMetadata,
            ToolObservation,
            TerminalOutcome,
        )
        
        # Parse dates
        started_at = datetime.fromisoformat(record["started_at"]) if record.get("started_at") else datetime.utcnow()
        ended_at = datetime.fromisoformat(record["ended_at"]) if record.get("ended_at") else None
        
        # Parse steps
        steps = []
        for step_data in record.get("steps", []):
            obs_data = step_data.get("observation", {})
            observation = ToolObservation(
                stdout=obs_data.get("stdout"),
                stderr=obs_data.get("stderr"),
                result=obs_data.get("result"),
                error=obs_data.get("error"),
                exit_code=obs_data.get("exit_code"),
            )
            
            step = WorkflowStep(
                step_index=step_data["step_index"],
                tool_name=step_data["tool_name"],
                tool_arguments=step_data.get("tool_arguments", {}),
                observation=observation,
                reasoning=step_data.get("reasoning"),
            )
            steps.append(step)
        
        # Parse outcome
        outcome = None
        if record.get("terminal_outcome"):
            out_data = record["terminal_outcome"]
            outcome = TerminalOutcome(
                success=out_data["success"],
                score=out_data["score"],
                explanation=out_data.get("explanation", ""),
                verifier_name=out_data.get("verifier_name", ""),
            )
        
        # Parse metadata
        metadata = WorkflowMetadata()
        if record.get("metadata"):
            meta_data = record["metadata"]
            metadata = WorkflowMetadata(
                total_tool_calls=meta_data.get("total_tool_calls", 0),
                total_latency_ms=meta_data.get("total_latency_ms", 0.0),
                tool_errors=meta_data.get("tool_errors", 0),
                schema_violations=meta_data.get("schema_violations", 0),
                extra=meta_data.get("extra", {}),
            )
        
        return WorkflowTrajectory(
            trajectory_id=record.get("trajectory_id", ""),
            user_request=record["user_request"],
            system_prompt=record.get("system_prompt", ""),
            steps=steps,
            final_answer=record.get("final_answer"),
            outcome=outcome,
            metadata=metadata,
            started_at=started_at,
            ended_at=ended_at,
            tags=record.get("tags", {}),
        )


# Convenience functions

def export_trajectories_jsonl(
    trajectories: list[WorkflowTrajectory],
    output_path: str | Path,
) -> int:
    """Export trajectories to JSONL file."""
    exporter = TrajectoryExporter()
    return exporter.export_jsonl(trajectories, output_path)


def load_trajectories_jsonl(input_path: str | Path) -> list[WorkflowTrajectory]:
    """Load trajectories from JSONL file."""
    loader = TrajectoryLoader()
    return loader.load_jsonl(input_path)


def print_dataset_stats(trajectories: list[WorkflowTrajectory]) -> None:
    """Compute and print dataset statistics."""
    stats = compute_dataset_stats(trajectories)
    print(stats)
