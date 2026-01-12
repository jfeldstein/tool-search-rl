"""Tests for export and statistics module."""

import pytest
import tempfile
import json
from pathlib import Path

from rlvr_tool_training.core.workflow import (
    WorkflowTrajectory,
    ToolObservation,
    TerminalOutcome,
)
from rlvr_tool_training.core.export import (
    TrajectoryExporter,
    TrajectoryLoader,
    TrajectoryDatasetStats,
    compute_dataset_stats,
    export_trajectories_jsonl,
    load_trajectories_jsonl,
)


def create_sample_trajectory(
    id_suffix: str = "1",
    success: bool = True,
    num_steps: int = 3,
) -> WorkflowTrajectory:
    """Create a sample trajectory for testing."""
    traj = WorkflowTrajectory(
        trajectory_id=f"test-{id_suffix}",
        user_request=f"Test request {id_suffix}",
        system_prompt="Test system prompt",
    )
    
    for i in range(num_steps):
        traj.add_step(
            tool_name=f"tool_{i}",
            tool_arguments={"arg": i},
            observation=ToolObservation(result=f"result_{i}"),
        )
    
    traj.metadata.total_latency_ms = 1000.0 * num_steps
    traj.final_answer = "Final answer"
    traj.outcome = TerminalOutcome(
        success=success,
        score=1.0 if success else 0.0,
        explanation="Test outcome",
        verifier_name="test",
    )
    
    return traj


class TestTrajectoryExporter:
    def test_export_jsonl(self):
        trajectories = [
            create_sample_trajectory("1"),
            create_sample_trajectory("2"),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"
            exporter = TrajectoryExporter()
            
            count = exporter.export_jsonl(trajectories, filepath)
            
            assert count == 2
            assert filepath.exists()
            
            # Verify content
            with open(filepath) as f:
                lines = f.readlines()
            assert len(lines) == 2
            
            # Each line should be valid JSON
            for line in lines:
                record = json.loads(line)
                assert "trajectory_id" in record
                assert "steps" in record
                assert "terminal_outcome" in record
    
    def test_export_jsonl_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"
            exporter = TrajectoryExporter()
            
            # First export
            exporter.export_jsonl([create_sample_trajectory("1")], filepath)
            
            # Append
            exporter.export_jsonl([create_sample_trajectory("2")], filepath, append=True)
            
            with open(filepath) as f:
                lines = f.readlines()
            assert len(lines) == 2
    
    def test_export_json(self):
        trajectories = [create_sample_trajectory("1")]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            exporter = TrajectoryExporter()
            
            count = exporter.export_json(trajectories, filepath)
            
            assert count == 1
            
            with open(filepath) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1


class TestTrajectoryLoader:
    def test_load_jsonl(self):
        trajectories = [
            create_sample_trajectory("1"),
            create_sample_trajectory("2", success=False),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"
            
            # Export first
            exporter = TrajectoryExporter()
            exporter.export_jsonl(trajectories, filepath)
            
            # Then load
            loader = TrajectoryLoader()
            loaded = loader.load_jsonl(filepath)
            
            assert len(loaded) == 2
            assert loaded[0].trajectory_id == "test-1"
            assert loaded[1].trajectory_id == "test-2"
            assert loaded[0].succeeded is True
            assert loaded[1].succeeded is False
    
    def test_load_json(self):
        trajectories = [create_sample_trajectory("1")]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            
            exporter = TrajectoryExporter()
            exporter.export_json(trajectories, filepath)
            
            loader = TrajectoryLoader()
            loaded = loader.load_json(filepath)
            
            assert len(loaded) == 1
            assert loaded[0].user_request == "Test request 1"
    
    def test_iter_jsonl(self):
        trajectories = [
            create_sample_trajectory(str(i)) for i in range(5)
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"
            
            exporter = TrajectoryExporter()
            exporter.export_jsonl(trajectories, filepath)
            
            loader = TrajectoryLoader()
            loaded = list(loader.iter_jsonl(filepath))
            
            assert len(loaded) == 5
    
    def test_roundtrip_preserves_data(self):
        original = create_sample_trajectory("1")
        original.tags = {"env": "test", "version": "1.0"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"
            
            exporter = TrajectoryExporter()
            exporter.export_jsonl([original], filepath)
            
            loader = TrajectoryLoader()
            loaded = loader.load_jsonl(filepath)[0]
            
            assert loaded.trajectory_id == original.trajectory_id
            assert loaded.user_request == original.user_request
            assert loaded.num_steps == original.num_steps
            assert loaded.final_answer == original.final_answer
            assert loaded.outcome.score == original.outcome.score
            assert loaded.tags == original.tags


class TestComputeDatasetStats:
    def test_empty_dataset(self):
        stats = compute_dataset_stats([])
        
        assert stats.total_trajectories == 0
        assert stats.success_rate == 0.0
    
    def test_all_successful(self):
        trajectories = [
            create_sample_trajectory(str(i), success=True)
            for i in range(5)
        ]
        
        stats = compute_dataset_stats(trajectories)
        
        assert stats.total_trajectories == 5
        assert stats.completed_trajectories == 5
        assert stats.successful_trajectories == 5
        assert stats.success_rate == 1.0
    
    def test_mixed_success(self):
        trajectories = [
            create_sample_trajectory("1", success=True),
            create_sample_trajectory("2", success=True),
            create_sample_trajectory("3", success=False),
        ]
        
        stats = compute_dataset_stats(trajectories)
        
        assert stats.total_trajectories == 3
        assert stats.successful_trajectories == 2
        assert stats.failed_trajectories == 1
        assert abs(stats.success_rate - 2/3) < 0.01
    
    def test_score_statistics(self):
        trajectories = [
            create_sample_trajectory("1", success=True),   # score=1.0
            create_sample_trajectory("2", success=False),  # score=0.0
        ]
        
        stats = compute_dataset_stats(trajectories)
        
        assert stats.score_mean == 0.5
        assert stats.score_min == 0.0
        assert stats.score_max == 1.0
    
    def test_step_statistics(self):
        trajectories = [
            create_sample_trajectory("1", num_steps=2),
            create_sample_trajectory("2", num_steps=4),
            create_sample_trajectory("3", num_steps=6),
        ]
        
        stats = compute_dataset_stats(trajectories)
        
        assert stats.min_steps == 2
        assert stats.max_steps == 6
        assert stats.avg_steps == 4.0
        assert stats.total_steps == 12
    
    def test_tool_usage_distribution(self):
        trajectories = [
            create_sample_trajectory("1", num_steps=3),  # tool_0, tool_1, tool_2
            create_sample_trajectory("2", num_steps=2),  # tool_0, tool_1
        ]
        
        stats = compute_dataset_stats(trajectories)
        
        assert stats.tool_usage["tool_0"] == 2
        assert stats.tool_usage["tool_1"] == 2
        assert stats.tool_usage["tool_2"] == 1
    
    def test_stats_to_dict(self):
        trajectories = [create_sample_trajectory("1")]
        stats = compute_dataset_stats(trajectories)
        
        data = stats.to_dict()
        
        assert "counts" in data
        assert "scores" in data
        assert "steps" in data
        assert "tool_usage" in data


class TestConvenienceFunctions:
    def test_export_and_load_jsonl(self):
        trajectories = [create_sample_trajectory("1")]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"
            
            count = export_trajectories_jsonl(trajectories, filepath)
            loaded = load_trajectories_jsonl(filepath)
            
            assert count == 1
            assert len(loaded) == 1
            assert loaded[0].trajectory_id == "test-1"
