"""
Terminal Verifiers Module

Provides a pluggable interface for verifiers that evaluate completed
workflow trajectories and produce terminal scores.

Verifiers take a completed trajectory and return:
- score (float): 0.0 to 1.0
- success (bool): pass/fail
- explanation (str): for debugging

This is the core mechanism for computing the sparse terminal reward
used in sequence-level learning.
"""

from __future__ import annotations

import subprocess
import re
import json
from abc import ABC, abstractmethod
from typing import Any, Callable
from pathlib import Path

from .workflow import WorkflowTrajectory, TerminalOutcome


class TerminalVerifier(ABC):
    """
    Abstract base class for terminal verifiers.
    
    A verifier evaluates a completed workflow trajectory and produces
    a terminal outcome (score + success + explanation).
    
    Implement this interface to create custom verifiers for your use case.
    
    Example:
        class MyVerifier(TerminalVerifier):
            @property
            def name(self) -> str:
                return "my_verifier"
            
            def verify(self, trajectory: WorkflowTrajectory) -> TerminalOutcome:
                # Your verification logic
                if some_condition:
                    return TerminalOutcome.passed("All checks passed", self.name)
                return TerminalOutcome.failed("Check failed", self.name)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this verifier."""
        pass
    
    @abstractmethod
    def verify(self, trajectory: WorkflowTrajectory) -> TerminalOutcome:
        """
        Verify a completed trajectory and return terminal outcome.
        
        Args:
            trajectory: The completed workflow trajectory to verify
            
        Returns:
            TerminalOutcome with score, success, and explanation
        """
        pass
    
    def __call__(self, trajectory: WorkflowTrajectory) -> TerminalOutcome:
        """Allow using verifier as a callable."""
        return self.verify(trajectory)


class PytestVerifier(TerminalVerifier):
    """
    Verifier that runs pytest and maps results to terminal score.
    
    - All tests pass → score=1.0, success=True
    - Some tests fail → score=passed/total, success=False
    - Pytest errors → score=0.0, success=False
    
    This is useful for verifying that code changes made by the agent
    result in passing tests.
    
    Example:
        verifier = PytestVerifier(test_path="tests/")
        outcome = verifier.verify(trajectory)
    """
    
    def __init__(
        self,
        test_path: str = "tests/",
        working_dir: str | None = None,
        timeout: int = 300,
        pytest_args: list[str] | None = None,
    ):
        """
        Args:
            test_path: Path to test directory or file
            working_dir: Working directory for pytest (defaults to cwd)
            timeout: Timeout in seconds
            pytest_args: Additional pytest arguments
        """
        self.test_path = test_path
        self.working_dir = working_dir
        self.timeout = timeout
        self.pytest_args = pytest_args or []
    
    @property
    def name(self) -> str:
        return "pytest"
    
    def verify(self, trajectory: WorkflowTrajectory) -> TerminalOutcome:
        """Run pytest and compute terminal score."""
        cmd = ["python", "-m", "pytest", self.test_path, "-v", "--tb=short"]
        cmd.extend(self.pytest_args)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            # Parse pytest output
            stdout = result.stdout
            stderr = result.stderr
            
            # Extract test counts from pytest output
            # Pattern: "X passed" or "X passed, Y failed"
            passed = 0
            failed = 0
            errors = 0
            
            passed_match = re.search(r"(\d+) passed", stdout)
            if passed_match:
                passed = int(passed_match.group(1))
            
            failed_match = re.search(r"(\d+) failed", stdout)
            if failed_match:
                failed = int(failed_match.group(1))
            
            error_match = re.search(r"(\d+) error", stdout)
            if error_match:
                errors = int(error_match.group(1))
            
            total = passed + failed + errors
            
            if total == 0:
                return TerminalOutcome(
                    success=False,
                    score=0.0,
                    explanation="No tests found or pytest failed to run",
                    verifier_name=self.name,
                    verifier_metadata={"stdout": stdout, "stderr": stderr, "exit_code": result.returncode}
                )
            
            score = passed / total if total > 0 else 0.0
            success = (failed == 0 and errors == 0)
            
            explanation = f"{passed}/{total} tests passed"
            if failed > 0:
                explanation += f", {failed} failed"
            if errors > 0:
                explanation += f", {errors} errors"
            
            return TerminalOutcome(
                success=success,
                score=score,
                explanation=explanation,
                verifier_name=self.name,
                verifier_metadata={
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "total": total,
                    "exit_code": result.returncode,
                }
            )
            
        except subprocess.TimeoutExpired:
            return TerminalOutcome(
                success=False,
                score=0.0,
                explanation=f"Pytest timed out after {self.timeout}s",
                verifier_name=self.name,
            )
        except Exception as e:
            return TerminalOutcome(
                success=False,
                score=0.0,
                explanation=f"Pytest execution error: {str(e)}",
                verifier_name=self.name,
            )


class ArtifactVerifier(TerminalVerifier):
    """
    Verifier that checks if expected artifacts were produced.
    
    Useful for verifying that the agent created/modified expected files
    or produced expected outputs.
    
    Example:
        verifier = ArtifactVerifier(
            required_files=["output.json", "report.md"],
            schema_validators={"output.json": json_schema}
        )
    """
    
    def __init__(
        self,
        required_files: list[str] | None = None,
        schema_validators: dict[str, dict] | None = None,
        content_checks: dict[str, Callable[[str], bool]] | None = None,
        base_path: str = ".",
    ):
        """
        Args:
            required_files: List of files that must exist
            schema_validators: JSON schema validators for specific files
            content_checks: Custom content validation functions
            base_path: Base path for file checks
        """
        self.required_files = required_files or []
        self.schema_validators = schema_validators or {}
        self.content_checks = content_checks or {}
        self.base_path = Path(base_path)
    
    @property
    def name(self) -> str:
        return "artifact"
    
    def verify(self, trajectory: WorkflowTrajectory) -> TerminalOutcome:
        """Verify that expected artifacts exist and are valid."""
        issues = []
        checks_passed = 0
        total_checks = len(self.required_files) + len(self.schema_validators) + len(self.content_checks)
        
        if total_checks == 0:
            return TerminalOutcome.passed("No artifact checks configured", self.name)
        
        # Check required files exist
        for filepath in self.required_files:
            full_path = self.base_path / filepath
            if full_path.exists():
                checks_passed += 1
            else:
                issues.append(f"Missing file: {filepath}")
        
        # Check JSON schema validators
        for filepath, schema in self.schema_validators.items():
            full_path = self.base_path / filepath
            if not full_path.exists():
                issues.append(f"Cannot validate schema, file missing: {filepath}")
                continue
            
            try:
                with open(full_path) as f:
                    data = json.load(f)
                
                # Basic schema validation (type checking)
                if self._validate_schema(data, schema):
                    checks_passed += 1
                else:
                    issues.append(f"Schema validation failed: {filepath}")
            except json.JSONDecodeError as e:
                issues.append(f"Invalid JSON in {filepath}: {e}")
            except Exception as e:
                issues.append(f"Error validating {filepath}: {e}")
        
        # Run custom content checks
        for filepath, check_fn in self.content_checks.items():
            full_path = self.base_path / filepath
            if not full_path.exists():
                issues.append(f"Cannot check content, file missing: {filepath}")
                continue
            
            try:
                content = full_path.read_text()
                if check_fn(content):
                    checks_passed += 1
                else:
                    issues.append(f"Content check failed: {filepath}")
            except Exception as e:
                issues.append(f"Error checking {filepath}: {e}")
        
        score = checks_passed / total_checks if total_checks > 0 else 0.0
        success = len(issues) == 0
        
        if success:
            explanation = f"All {total_checks} artifact checks passed"
        else:
            explanation = f"{checks_passed}/{total_checks} checks passed. Issues: {'; '.join(issues)}"
        
        return TerminalOutcome(
            success=success,
            score=score,
            explanation=explanation,
            verifier_name=self.name,
            verifier_metadata={
                "checks_passed": checks_passed,
                "total_checks": total_checks,
                "issues": issues,
            }
        )
    
    def _validate_schema(self, data: Any, schema: dict) -> bool:
        """Basic JSON schema validation (type checking only)."""
        schema_type = schema.get("type")
        
        if schema_type == "object":
            if not isinstance(data, dict):
                return False
            # Check required properties
            required = schema.get("required", [])
            for prop in required:
                if prop not in data:
                    return False
            return True
        elif schema_type == "array":
            return isinstance(data, list)
        elif schema_type == "string":
            return isinstance(data, str)
        elif schema_type == "number":
            return isinstance(data, (int, float))
        elif schema_type == "boolean":
            return isinstance(data, bool)
        
        return True  # Unknown type, assume valid


class FunctionVerifier(TerminalVerifier):
    """
    Verifier that uses a custom function for verification.
    
    Useful for quick custom verifiers without creating a full class.
    
    Example:
        def my_check(trajectory: WorkflowTrajectory) -> tuple[bool, float, str]:
            success = trajectory.final_answer is not None
            return success, 1.0 if success else 0.0, "Check complete"
        
        verifier = FunctionVerifier("my_check", my_check)
    """
    
    def __init__(
        self,
        verifier_name: str,
        verify_fn: Callable[[WorkflowTrajectory], tuple[bool, float, str]],
    ):
        """
        Args:
            verifier_name: Name for this verifier
            verify_fn: Function that takes trajectory and returns (success, score, explanation)
        """
        self._name = verifier_name
        self.verify_fn = verify_fn
    
    @property
    def name(self) -> str:
        return self._name
    
    def verify(self, trajectory: WorkflowTrajectory) -> TerminalOutcome:
        """Run the custom verification function."""
        try:
            success, score, explanation = self.verify_fn(trajectory)
            return TerminalOutcome(
                success=success,
                score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
                explanation=explanation,
                verifier_name=self.name,
            )
        except Exception as e:
            return TerminalOutcome(
                success=False,
                score=0.0,
                explanation=f"Verifier error: {str(e)}",
                verifier_name=self.name,
            )


class CompositeVerifier(TerminalVerifier):
    """
    Combines multiple verifiers with configurable aggregation.
    
    Aggregation modes:
    - "all": All verifiers must pass (AND)
    - "any": Any verifier must pass (OR)
    - "weighted": Weighted average of scores
    
    Example:
        verifier = CompositeVerifier(
            verifiers=[pytest_verifier, artifact_verifier],
            mode="all"
        )
    """
    
    def __init__(
        self,
        verifiers: list[TerminalVerifier],
        mode: str = "all",
        weights: list[float] | None = None,
    ):
        """
        Args:
            verifiers: List of verifiers to combine
            mode: Aggregation mode ("all", "any", "weighted")
            weights: Weights for "weighted" mode (must sum to 1.0)
        """
        self.verifiers = verifiers
        self.mode = mode
        self.weights = weights
        
        if mode == "weighted":
            if weights is None:
                self.weights = [1.0 / len(verifiers)] * len(verifiers)
            else:
                total = sum(weights)
                self.weights = [w / total for w in weights]
    
    @property
    def name(self) -> str:
        return f"composite({self.mode})"
    
    def verify(self, trajectory: WorkflowTrajectory) -> TerminalOutcome:
        """Run all verifiers and aggregate results."""
        results = [v.verify(trajectory) for v in self.verifiers]
        
        if self.mode == "all":
            success = all(r.success for r in results)
            score = min(r.score for r in results) if results else 0.0
        elif self.mode == "any":
            success = any(r.success for r in results)
            score = max(r.score for r in results) if results else 0.0
        elif self.mode == "weighted":
            success = sum(r.score * w for r, w in zip(results, self.weights)) >= 0.5
            score = sum(r.score * w for r, w in zip(results, self.weights))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        explanations = [f"{r.verifier_name}: {r.explanation}" for r in results]
        
        return TerminalOutcome(
            success=success,
            score=score,
            explanation=" | ".join(explanations),
            verifier_name=self.name,
            verifier_metadata={
                "individual_results": [
                    {"verifier": r.verifier_name, "success": r.success, "score": r.score}
                    for r in results
                ]
            }
        )


class WorkflowCompletionVerifier(TerminalVerifier):
    """
    Basic verifier that checks workflow completion status.
    
    Checks:
    - Trajectory has a final answer
    - No tool errors occurred
    - Completed within max steps
    
    This is a simple baseline verifier suitable for this repo.
    """
    
    def __init__(
        self,
        require_final_answer: bool = True,
        max_steps: int | None = None,
        max_errors: int = 0,
    ):
        self.require_final_answer = require_final_answer
        self.max_steps = max_steps
        self.max_errors = max_errors
    
    @property
    def name(self) -> str:
        return "workflow_completion"
    
    def verify(self, trajectory: WorkflowTrajectory) -> TerminalOutcome:
        """Verify basic workflow completion."""
        issues = []
        
        # Check final answer
        if self.require_final_answer and not trajectory.final_answer:
            issues.append("No final answer provided")
        
        # Check max steps
        if self.max_steps and trajectory.num_steps > self.max_steps:
            issues.append(f"Exceeded max steps ({trajectory.num_steps} > {self.max_steps})")
        
        # Check errors
        if trajectory.metadata.tool_errors > self.max_errors:
            issues.append(f"Too many tool errors ({trajectory.metadata.tool_errors} > {self.max_errors})")
        
        success = len(issues) == 0
        
        # Compute score based on completion quality
        score = 1.0
        if not trajectory.final_answer:
            score -= 0.3
        if self.max_steps and trajectory.num_steps > self.max_steps:
            score -= 0.2
        if trajectory.metadata.tool_errors > 0:
            score -= 0.1 * trajectory.metadata.tool_errors
        score = max(0.0, score)
        
        explanation = "Workflow completed successfully" if success else f"Issues: {'; '.join(issues)}"
        
        return TerminalOutcome(
            success=success,
            score=score,
            explanation=explanation,
            verifier_name=self.name,
            verifier_metadata={
                "has_final_answer": bool(trajectory.final_answer),
                "num_steps": trajectory.num_steps,
                "tool_errors": trajectory.metadata.tool_errors,
            }
        )
