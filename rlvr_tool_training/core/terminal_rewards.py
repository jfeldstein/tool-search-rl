"""
Terminal Rewards Module

Provides sequence-level reward aggregation with optional shaping penalties.

Key principle: The core signal is a SPARSE TERMINAL REWARD assigned at the
end of the trajectory. We do NOT label individual steps.

Shaping penalties are applied at the trajectory level (not per-step) to
discourage undesirable patterns like:
- Too many tool calls
- Excessive latency
- Tool schema/usage errors

This supports RL-style returns or preference-style training where we
cannot identify which specific tool call was "right."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .workflow import WorkflowTrajectory, TerminalOutcome
from .verifiers import TerminalVerifier


@dataclass
class ShapingPenalty:
    """
    A trajectory-level shaping penalty.
    
    Penalties reduce the terminal score based on trajectory characteristics.
    They do NOT create per-step labels - just adjust the final score.
    """
    name: str
    penalty_fn: Callable[[WorkflowTrajectory], float]
    max_penalty: float = 0.5  # Maximum penalty this can apply
    enabled: bool = True
    
    def compute(self, trajectory: WorkflowTrajectory) -> float:
        """Compute penalty for trajectory (returns value between 0 and max_penalty)."""
        if not self.enabled:
            return 0.0
        penalty = self.penalty_fn(trajectory)
        return min(max(0.0, penalty), self.max_penalty)


def too_many_calls_penalty(
    max_calls: int = 10,
    penalty_per_extra: float = 0.05,
) -> ShapingPenalty:
    """
    Penalty for using too many tool calls.
    
    Encourages efficient tool use without requiring per-step labels.
    """
    def compute(trajectory: WorkflowTrajectory) -> float:
        extra_calls = max(0, trajectory.num_steps - max_calls)
        return extra_calls * penalty_per_extra
    
    return ShapingPenalty(
        name="too_many_calls",
        penalty_fn=compute,
        max_penalty=0.3,
    )


def excessive_latency_penalty(
    max_latency_ms: float = 60000,  # 60 seconds
    penalty_per_second: float = 0.01,
) -> ShapingPenalty:
    """
    Penalty for excessive total latency.
    
    Encourages faster execution without per-step timing requirements.
    """
    def compute(trajectory: WorkflowTrajectory) -> float:
        if trajectory.metadata.total_latency_ms <= max_latency_ms:
            return 0.0
        extra_ms = trajectory.metadata.total_latency_ms - max_latency_ms
        extra_seconds = extra_ms / 1000
        return extra_seconds * penalty_per_second
    
    return ShapingPenalty(
        name="excessive_latency",
        penalty_fn=compute,
        max_penalty=0.2,
    )


def tool_error_penalty(
    penalty_per_error: float = 0.1,
) -> ShapingPenalty:
    """
    Penalty for tool execution errors.
    
    Encourages successful tool usage without labeling which calls failed.
    """
    def compute(trajectory: WorkflowTrajectory) -> float:
        return trajectory.metadata.tool_errors * penalty_per_error
    
    return ShapingPenalty(
        name="tool_errors",
        penalty_fn=compute,
        max_penalty=0.4,
    )


def schema_violation_penalty(
    penalty_per_violation: float = 0.15,
) -> ShapingPenalty:
    """
    Penalty for schema/usage violations.
    
    Encourages proper tool usage format.
    """
    def compute(trajectory: WorkflowTrajectory) -> float:
        return trajectory.metadata.schema_violations * penalty_per_violation
    
    return ShapingPenalty(
        name="schema_violations",
        penalty_fn=compute,
        max_penalty=0.3,
    )


def no_final_answer_penalty(
    penalty: float = 0.2,
) -> ShapingPenalty:
    """
    Penalty for not providing a final answer.
    """
    def compute(trajectory: WorkflowTrajectory) -> float:
        return penalty if not trajectory.final_answer else 0.0
    
    return ShapingPenalty(
        name="no_final_answer",
        penalty_fn=compute,
        max_penalty=penalty,
    )


@dataclass
class TerminalRewardConfig:
    """
    Configuration for terminal reward computation.
    
    Controls how the final trajectory-level reward is computed:
    - Base score from verifier
    - Shaping penalties applied
    - Score bounds
    """
    # Shaping penalties to apply
    penalties: list[ShapingPenalty] = field(default_factory=list)
    
    # Score bounds
    min_score: float = 0.0
    max_score: float = 1.0
    
    # Whether to clamp penalties to not go below min_score
    clamp_penalties: bool = True
    
    @classmethod
    def default(cls) -> TerminalRewardConfig:
        """Create default config with standard penalties."""
        return cls(
            penalties=[
                too_many_calls_penalty(),
                excessive_latency_penalty(),
                tool_error_penalty(),
            ]
        )
    
    @classmethod
    def strict(cls) -> TerminalRewardConfig:
        """Create strict config with all penalties."""
        return cls(
            penalties=[
                too_many_calls_penalty(max_calls=5, penalty_per_extra=0.1),
                excessive_latency_penalty(max_latency_ms=30000),
                tool_error_penalty(penalty_per_error=0.2),
                schema_violation_penalty(),
                no_final_answer_penalty(penalty=0.3),
            ]
        )
    
    @classmethod
    def minimal(cls) -> TerminalRewardConfig:
        """Create minimal config with no shaping penalties."""
        return cls(penalties=[])


@dataclass
class TerminalRewardResult:
    """
    Result of computing terminal reward for a trajectory.
    
    Contains:
    - Final score (after all penalties)
    - Base score (from verifier)
    - Breakdown of penalties applied
    - Success status
    """
    final_score: float
    base_score: float
    success: bool
    explanation: str
    
    # Penalty breakdown
    penalties_applied: dict[str, float] = field(default_factory=dict)
    total_penalty: float = 0.0
    
    # Source information
    verifier_name: str = ""
    verifier_explanation: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_score": self.final_score,
            "base_score": self.base_score,
            "success": self.success,
            "explanation": self.explanation,
            "penalties_applied": self.penalties_applied,
            "total_penalty": self.total_penalty,
            "verifier_name": self.verifier_name,
            "verifier_explanation": self.verifier_explanation,
        }


class TerminalRewardComputer:
    """
    Computes the terminal reward for a workflow trajectory.
    
    This is the main interface for sequence-level reward computation:
    
    1. Run verifier to get base score
    2. Apply trajectory-level shaping penalties
    3. Return final terminal reward
    
    The reward is SPARSE and TERMINAL - it summarizes the entire trajectory
    in a single score, without labeling individual steps.
    
    Example:
        verifier = PytestVerifier(test_path="tests/")
        config = TerminalRewardConfig.default()
        
        computer = TerminalRewardComputer(verifier, config)
        result = computer.compute(trajectory)
        
        print(f"Terminal reward: {result.final_score}")
    """
    
    def __init__(
        self,
        verifier: TerminalVerifier,
        config: TerminalRewardConfig | None = None,
    ):
        """
        Args:
            verifier: The terminal verifier to use
            config: Reward configuration (defaults to standard config)
        """
        self.verifier = verifier
        self.config = config or TerminalRewardConfig.default()
    
    def compute(self, trajectory: WorkflowTrajectory) -> TerminalRewardResult:
        """
        Compute the terminal reward for a trajectory.
        
        Returns:
            TerminalRewardResult with final score and breakdown
        """
        # Step 1: Get base score from verifier
        outcome = self.verifier.verify(trajectory)
        base_score = outcome.score
        
        # Step 2: Apply shaping penalties
        penalties_applied = {}
        total_penalty = 0.0
        
        for penalty in self.config.penalties:
            if penalty.enabled:
                penalty_value = penalty.compute(trajectory)
                if penalty_value > 0:
                    penalties_applied[penalty.name] = penalty_value
                    total_penalty += penalty_value
        
        # Step 3: Compute final score
        final_score = base_score - total_penalty
        
        # Clamp to bounds
        if self.config.clamp_penalties:
            final_score = max(self.config.min_score, min(self.config.max_score, final_score))
        
        # Determine success (based on final score, not just verifier)
        success = outcome.success and final_score >= 0.5
        
        # Build explanation
        explanation_parts = [outcome.explanation]
        if penalties_applied:
            penalty_strs = [f"{k}={v:.2f}" for k, v in penalties_applied.items()]
            explanation_parts.append(f"Penalties: {', '.join(penalty_strs)}")
        
        return TerminalRewardResult(
            final_score=final_score,
            base_score=base_score,
            success=success,
            explanation=" | ".join(explanation_parts),
            penalties_applied=penalties_applied,
            total_penalty=total_penalty,
            verifier_name=outcome.verifier_name,
            verifier_explanation=outcome.explanation,
        )
    
    def compute_and_attach(self, trajectory: WorkflowTrajectory) -> TerminalRewardResult:
        """
        Compute terminal reward and attach outcome to trajectory.
        
        This modifies the trajectory in place, setting its outcome field.
        """
        result = self.compute(trajectory)
        
        # Create outcome from result
        outcome = TerminalOutcome(
            success=result.success,
            score=result.final_score,
            explanation=result.explanation,
            verifier_name=result.verifier_name,
            verifier_metadata={
                "base_score": result.base_score,
                "penalties_applied": result.penalties_applied,
                "total_penalty": result.total_penalty,
            }
        )
        
        trajectory.outcome = outcome
        return result


def compute_terminal_reward(
    trajectory: WorkflowTrajectory,
    verifier: TerminalVerifier,
    config: TerminalRewardConfig | None = None,
) -> TerminalRewardResult:
    """
    Convenience function to compute terminal reward.
    
    Args:
        trajectory: The workflow trajectory to evaluate
        verifier: The verifier to use
        config: Optional reward config
    
    Returns:
        TerminalRewardResult with final score
    """
    computer = TerminalRewardComputer(verifier, config)
    return computer.compute(trajectory)
