"""Tests for workflow trajectory module."""

import pytest
from datetime import datetime, timedelta

from rlvr_tool_training.core.workflow import (
    WorkflowTrajectory,
    WorkflowStep,
    WorkflowMetadata,
    ToolObservation,
    TerminalOutcome,
    WorkflowRecorder,
)


class TestToolObservation:
    def test_success_when_no_error(self):
        obs = ToolObservation(result="test")
        assert obs.success is True
    
    def test_failure_on_error(self):
        obs = ToolObservation(error="Something went wrong")
        assert obs.success is False
    
    def test_failure_on_exit_code(self):
        obs = ToolObservation(exit_code=1)
        assert obs.success is False
    
    def test_success_on_zero_exit_code(self):
        obs = ToolObservation(exit_code=0)
        assert obs.success is True


class TestWorkflowStep:
    def test_basic_step(self):
        step = WorkflowStep(
            step_index=0,
            tool_name="read_file",
            tool_arguments={"path": "test.py"},
        )
        assert step.tool_name == "read_file"
        assert step.step_index == 0
    
    def test_step_with_observation(self):
        step = WorkflowStep(
            step_index=0,
            tool_name="run_command",
            tool_arguments={"command": "ls"},
            observation=ToolObservation(stdout="file1\nfile2", exit_code=0),
        )
        assert step.succeeded is True
        assert step.observation.stdout == "file1\nfile2"
    
    def test_duration_calculation(self):
        start = datetime.utcnow()
        end = start + timedelta(milliseconds=500)
        
        step = WorkflowStep(
            step_index=0,
            tool_name="slow_tool",
            tool_arguments={},
            started_at=start,
            ended_at=end,
        )
        
        assert step.duration_ms is not None
        assert abs(step.duration_ms - 500) < 1


class TestTerminalOutcome:
    def test_passed(self):
        outcome = TerminalOutcome.passed("All tests passed", "pytest")
        assert outcome.success is True
        assert outcome.score == 1.0
        assert outcome.verifier_name == "pytest"
    
    def test_failed(self):
        outcome = TerminalOutcome.failed("Tests failed", "pytest")
        assert outcome.success is False
        assert outcome.score == 0.0
    
    def test_partial(self):
        outcome = TerminalOutcome.partial(0.7, "Most tests passed")
        assert outcome.success is True  # 0.7 >= 0.5
        assert outcome.score == 0.7
    
    def test_partial_below_threshold(self):
        outcome = TerminalOutcome.partial(0.3, "Few tests passed")
        assert outcome.success is False  # 0.3 < 0.5


class TestWorkflowTrajectory:
    def test_basic_trajectory(self):
        traj = WorkflowTrajectory(user_request="Fix the bug")
        assert traj.user_request == "Fix the bug"
        assert traj.num_steps == 0
        assert not traj.is_complete
    
    def test_add_step(self):
        traj = WorkflowTrajectory(user_request="Fix the bug")
        
        step = traj.add_step(
            tool_name="read_file",
            tool_arguments={"path": "main.py"},
            observation=ToolObservation(result="file contents"),
        )
        
        assert traj.num_steps == 1
        assert step.step_index == 0
        assert traj.metadata.total_tool_calls == 1
    
    def test_add_multiple_steps(self):
        traj = WorkflowTrajectory(user_request="Fix the bug")
        
        traj.add_step("read_file", {"path": "main.py"})
        traj.add_step("edit_file", {"path": "main.py", "content": "new"})
        traj.add_step("run_tests", {})
        
        assert traj.num_steps == 3
        assert traj.steps[0].step_index == 0
        assert traj.steps[1].step_index == 1
        assert traj.steps[2].step_index == 2
    
    def test_complete_trajectory(self):
        traj = WorkflowTrajectory(user_request="Fix the bug")
        traj.add_step("read_file", {"path": "main.py"})
        
        traj.complete(
            final_answer="Fixed the bug",
            outcome=TerminalOutcome.passed("All tests pass"),
        )
        
        assert traj.is_complete
        assert traj.succeeded
        assert traj.final_answer == "Fixed the bug"
        assert traj.terminal_score == 1.0
    
    def test_error_tracking(self):
        traj = WorkflowTrajectory(user_request="Test errors")
        
        # Add a failing step
        traj.add_step(
            "bad_command",
            {},
            observation=ToolObservation(error="Command not found"),
        )
        
        assert traj.metadata.tool_errors == 1
    
    def test_serialization_roundtrip(self):
        traj = WorkflowTrajectory(
            user_request="Test serialization",
            system_prompt="You are helpful",
        )
        traj.add_step("tool1", {"arg": "value"})
        traj.complete("Done", TerminalOutcome.passed())
        
        # To dict and back
        data = traj.to_dict()
        restored = WorkflowTrajectory.from_dict(data)
        
        assert restored.user_request == traj.user_request
        assert restored.num_steps == traj.num_steps
        assert restored.succeeded == traj.succeeded


class TestWorkflowRecorder:
    def test_record_trajectory(self):
        recorder = WorkflowRecorder()
        
        with recorder.start_trajectory("Test request") as ctx:
            with ctx.record_step("tool1", {"x": 1}) as step:
                step.set_result("result1")
            
            with ctx.record_step("tool2", {"y": 2}) as step:
                step.set_result("result2")
            
            ctx.set_final_answer("Done")
        
        assert len(recorder.trajectories) == 1
        traj = recorder.trajectories[0]
        assert traj.num_steps == 2
        assert traj.final_answer == "Done"
    
    def test_step_error_handling(self):
        recorder = WorkflowRecorder()
        
        with recorder.start_trajectory("Test errors") as ctx:
            with ctx.record_step("failing_tool", {}) as step:
                step.set_error("Something went wrong")
        
        traj = recorder.trajectories[0]
        assert traj.steps[0].observation.error == "Something went wrong"
