"""Tests for terminal rewards module."""

import pytest

from rlvr_tool_training.core.workflow import (
    WorkflowTrajectory,
    WorkflowMetadata,
    ToolObservation,
    TerminalOutcome,
)
from rlvr_tool_training.core.verifiers import FunctionVerifier
from rlvr_tool_training.core.terminal_rewards import (
    TerminalRewardComputer,
    TerminalRewardConfig,
    ShapingPenalty,
    too_many_calls_penalty,
    excessive_latency_penalty,
    tool_error_penalty,
    compute_terminal_reward,
)


def create_trajectory_with_metadata(
    num_steps: int = 5,
    latency_ms: float = 10000,
    errors: int = 0,
    has_answer: bool = True,
) -> WorkflowTrajectory:
    """Helper to create trajectories with specific metadata."""
    traj = WorkflowTrajectory(user_request="Test")
    
    for i in range(num_steps):
        traj.add_step(f"tool_{i}", {"arg": i})
    
    traj.metadata.total_latency_ms = latency_ms
    traj.metadata.tool_errors = errors
    
    if has_answer:
        traj.final_answer = "Answer"
    
    return traj


class TestShapingPenalties:
    def test_too_many_calls_no_penalty(self):
        penalty = too_many_calls_penalty(max_calls=10)
        traj = create_trajectory_with_metadata(num_steps=5)
        
        value = penalty.compute(traj)
        
        assert value == 0.0
    
    def test_too_many_calls_with_penalty(self):
        penalty = too_many_calls_penalty(max_calls=5, penalty_per_extra=0.1)
        traj = create_trajectory_with_metadata(num_steps=8)
        
        value = penalty.compute(traj)
        
        # 8 - 5 = 3 extra calls * 0.1 = 0.3
        assert abs(value - 0.3) < 0.01
    
    def test_excessive_latency_no_penalty(self):
        penalty = excessive_latency_penalty(max_latency_ms=60000)
        traj = create_trajectory_with_metadata(latency_ms=30000)
        
        value = penalty.compute(traj)
        
        assert value == 0.0
    
    def test_excessive_latency_with_penalty(self):
        penalty = excessive_latency_penalty(max_latency_ms=10000, penalty_per_second=0.02)
        traj = create_trajectory_with_metadata(latency_ms=20000)
        
        value = penalty.compute(traj)
        
        # 20000 - 10000 = 10000ms extra = 10s * 0.02 = 0.2
        assert abs(value - 0.2) < 0.01
    
    def test_tool_error_penalty(self):
        penalty = tool_error_penalty(penalty_per_error=0.1)
        traj = create_trajectory_with_metadata(errors=3)
        
        value = penalty.compute(traj)
        
        assert abs(value - 0.3) < 0.01
    
    def test_penalty_max_cap(self):
        penalty = ShapingPenalty(
            name="test",
            penalty_fn=lambda t: 1.0,  # Would return 1.0
            max_penalty=0.3,
        )
        traj = create_trajectory_with_metadata()
        
        value = penalty.compute(traj)
        
        assert value == 0.3  # Capped at max
    
    def test_disabled_penalty(self):
        penalty = too_many_calls_penalty(max_calls=1)
        penalty.enabled = False
        traj = create_trajectory_with_metadata(num_steps=100)
        
        value = penalty.compute(traj)
        
        assert value == 0.0


class TestTerminalRewardConfig:
    def test_default_config(self):
        config = TerminalRewardConfig.default()
        
        assert len(config.penalties) > 0
        assert config.min_score == 0.0
        assert config.max_score == 1.0
    
    def test_strict_config(self):
        config = TerminalRewardConfig.strict()
        
        assert len(config.penalties) >= 4
    
    def test_minimal_config(self):
        config = TerminalRewardConfig.minimal()
        
        assert len(config.penalties) == 0


class TestTerminalRewardComputer:
    def test_perfect_trajectory(self):
        verifier = FunctionVerifier("test", lambda t: (True, 1.0, "pass"))
        config = TerminalRewardConfig.minimal()  # No penalties
        
        computer = TerminalRewardComputer(verifier, config)
        traj = create_trajectory_with_metadata(num_steps=3)
        
        result = computer.compute(traj)
        
        assert result.final_score == 1.0
        assert result.success is True
        assert result.total_penalty == 0.0
    
    def test_with_penalties(self):
        verifier = FunctionVerifier("test", lambda t: (True, 1.0, "pass"))
        config = TerminalRewardConfig(penalties=[
            too_many_calls_penalty(max_calls=3, penalty_per_extra=0.1)
        ])
        
        computer = TerminalRewardComputer(verifier, config)
        traj = create_trajectory_with_metadata(num_steps=5)  # 2 extra calls
        
        result = computer.compute(traj)
        
        assert result.base_score == 1.0
        assert result.total_penalty == 0.2
        assert result.final_score == 0.8
        assert "too_many_calls" in result.penalties_applied
    
    def test_failed_verification(self):
        verifier = FunctionVerifier("test", lambda t: (False, 0.0, "fail"))
        config = TerminalRewardConfig.minimal()
        
        computer = TerminalRewardComputer(verifier, config)
        traj = create_trajectory_with_metadata()
        
        result = computer.compute(traj)
        
        assert result.final_score == 0.0
        assert result.success is False
    
    def test_compute_and_attach(self):
        verifier = FunctionVerifier("test", lambda t: (True, 0.8, "partial"))
        config = TerminalRewardConfig.minimal()
        
        computer = TerminalRewardComputer(verifier, config)
        traj = create_trajectory_with_metadata()
        
        result = computer.compute_and_attach(traj)
        
        assert traj.outcome is not None
        assert traj.outcome.score == 0.8
        assert traj.terminal_score == 0.8
    
    def test_score_clamping(self):
        verifier = FunctionVerifier("test", lambda t: (True, 0.5, "partial"))
        config = TerminalRewardConfig(
            penalties=[
                ShapingPenalty("big_penalty", lambda t: 1.0, max_penalty=1.0)
            ],
            clamp_penalties=True,
        )
        
        computer = TerminalRewardComputer(verifier, config)
        traj = create_trajectory_with_metadata()
        
        result = computer.compute(traj)
        
        # 0.5 - 1.0 = -0.5, but clamped to 0.0
        assert result.final_score == 0.0


class TestComputeTerminalReward:
    def test_convenience_function(self):
        verifier = FunctionVerifier("test", lambda t: (True, 1.0, "pass"))
        traj = create_trajectory_with_metadata()
        
        result = compute_terminal_reward(traj, verifier)
        
        assert result.final_score == 1.0
        assert result.verifier_name == "test"
