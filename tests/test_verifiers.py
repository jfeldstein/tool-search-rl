"""Tests for terminal verifiers module."""

import pytest
import tempfile
import json
from pathlib import Path

from rlvr_tool_training.core.workflow import (
    WorkflowTrajectory,
    WorkflowStep,
    ToolObservation,
    TerminalOutcome,
)
from rlvr_tool_training.core.verifiers import (
    TerminalVerifier,
    WorkflowCompletionVerifier,
    FunctionVerifier,
    CompositeVerifier,
    ArtifactVerifier,
)


def create_test_trajectory(
    num_steps: int = 3,
    has_answer: bool = True,
    errors: int = 0,
) -> WorkflowTrajectory:
    """Helper to create test trajectories."""
    traj = WorkflowTrajectory(user_request="Test request")
    
    for i in range(num_steps):
        has_error = i < errors
        traj.add_step(
            tool_name=f"tool_{i}",
            tool_arguments={"arg": i},
            observation=ToolObservation(
                result=f"result_{i}" if not has_error else None,
                error="error" if has_error else None,
            ),
        )
    
    if has_answer:
        traj.final_answer = "Final answer"
    
    return traj


class TestWorkflowCompletionVerifier:
    def test_successful_trajectory(self):
        verifier = WorkflowCompletionVerifier()
        traj = create_test_trajectory(num_steps=3, has_answer=True, errors=0)
        
        outcome = verifier.verify(traj)
        
        assert outcome.success is True
        assert outcome.score == 1.0
        assert "successfully" in outcome.explanation.lower()
    
    def test_missing_answer(self):
        verifier = WorkflowCompletionVerifier(require_final_answer=True)
        traj = create_test_trajectory(has_answer=False)
        
        outcome = verifier.verify(traj)
        
        assert outcome.success is False
        assert "final answer" in outcome.explanation.lower()
    
    def test_exceeds_max_steps(self):
        verifier = WorkflowCompletionVerifier(max_steps=5)
        traj = create_test_trajectory(num_steps=10)
        
        outcome = verifier.verify(traj)
        
        assert outcome.success is False
        assert "max steps" in outcome.explanation.lower()
    
    def test_tool_errors(self):
        verifier = WorkflowCompletionVerifier(max_errors=0)
        traj = create_test_trajectory(errors=2)
        
        outcome = verifier.verify(traj)
        
        assert outcome.success is False
        assert "errors" in outcome.explanation.lower()


class TestFunctionVerifier:
    def test_custom_verifier_pass(self):
        def my_check(traj):
            has_answer = traj.final_answer is not None
            return has_answer, 1.0 if has_answer else 0.0, "Check complete"
        
        verifier = FunctionVerifier("my_check", my_check)
        traj = create_test_trajectory(has_answer=True)
        
        outcome = verifier.verify(traj)
        
        assert outcome.success is True
        assert outcome.score == 1.0
        assert verifier.name == "my_check"
    
    def test_custom_verifier_fail(self):
        def my_check(traj):
            return False, 0.0, "Always fails"
        
        verifier = FunctionVerifier("fail_check", my_check)
        traj = create_test_trajectory()
        
        outcome = verifier.verify(traj)
        
        assert outcome.success is False
        assert outcome.score == 0.0
    
    def test_verifier_error_handling(self):
        def broken_check(traj):
            raise ValueError("Verifier crashed")
        
        verifier = FunctionVerifier("broken", broken_check)
        traj = create_test_trajectory()
        
        outcome = verifier.verify(traj)
        
        assert outcome.success is False
        assert "error" in outcome.explanation.lower()


class TestCompositeVerifier:
    def test_all_mode_all_pass(self):
        v1 = FunctionVerifier("v1", lambda t: (True, 1.0, "pass"))
        v2 = FunctionVerifier("v2", lambda t: (True, 0.8, "pass"))
        
        composite = CompositeVerifier([v1, v2], mode="all")
        traj = create_test_trajectory()
        
        outcome = composite.verify(traj)
        
        assert outcome.success is True
        assert outcome.score == 0.8  # min of scores in "all" mode
    
    def test_all_mode_one_fail(self):
        v1 = FunctionVerifier("v1", lambda t: (True, 1.0, "pass"))
        v2 = FunctionVerifier("v2", lambda t: (False, 0.0, "fail"))
        
        composite = CompositeVerifier([v1, v2], mode="all")
        traj = create_test_trajectory()
        
        outcome = composite.verify(traj)
        
        assert outcome.success is False
    
    def test_any_mode(self):
        v1 = FunctionVerifier("v1", lambda t: (False, 0.0, "fail"))
        v2 = FunctionVerifier("v2", lambda t: (True, 1.0, "pass"))
        
        composite = CompositeVerifier([v1, v2], mode="any")
        traj = create_test_trajectory()
        
        outcome = composite.verify(traj)
        
        assert outcome.success is True
        assert outcome.score == 1.0  # max of scores in "any" mode
    
    def test_weighted_mode(self):
        v1 = FunctionVerifier("v1", lambda t: (True, 1.0, "pass"))
        v2 = FunctionVerifier("v2", lambda t: (True, 0.5, "partial"))
        
        composite = CompositeVerifier([v1, v2], mode="weighted", weights=[0.8, 0.2])
        traj = create_test_trajectory()
        
        outcome = composite.verify(traj)
        
        # Weighted: 1.0 * 0.8 + 0.5 * 0.2 = 0.9
        assert abs(outcome.score - 0.9) < 0.01


class TestArtifactVerifier:
    def test_required_files_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create required files
            (Path(tmpdir) / "output.json").write_text('{"status": "ok"}')
            (Path(tmpdir) / "report.txt").write_text("Report content")
            
            verifier = ArtifactVerifier(
                required_files=["output.json", "report.txt"],
                base_path=tmpdir,
            )
            traj = create_test_trajectory()
            
            outcome = verifier.verify(traj)
            
            assert outcome.success is True
    
    def test_required_files_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only create one file
            (Path(tmpdir) / "output.json").write_text('{}')
            
            verifier = ArtifactVerifier(
                required_files=["output.json", "missing.txt"],
                base_path=tmpdir,
            )
            traj = create_test_trajectory()
            
            outcome = verifier.verify(traj)
            
            assert outcome.success is False
            assert "missing" in outcome.explanation.lower()
    
    def test_schema_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid JSON
            (Path(tmpdir) / "data.json").write_text('{"name": "test", "count": 5}')
            
            verifier = ArtifactVerifier(
                schema_validators={
                    "data.json": {"type": "object", "required": ["name"]}
                },
                base_path=tmpdir,
            )
            traj = create_test_trajectory()
            
            outcome = verifier.verify(traj)
            
            assert outcome.success is True
    
    def test_content_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "output.txt").write_text("SUCCESS: All done")
            
            verifier = ArtifactVerifier(
                content_checks={
                    "output.txt": lambda c: "SUCCESS" in c
                },
                base_path=tmpdir,
            )
            traj = create_test_trajectory()
            
            outcome = verifier.verify(traj)
            
            assert outcome.success is True
