"""
Workflow Trajectory Module

Defines data structures for complete multi-step tool-use trajectories:
- User request → tool calls/args → tool observations/results → final answer
- Terminal outcome with single workflow-level score
- Metadata for cost, latency, error tracking

This supports sequence-level learning where the reward signal is sparse
and terminal - we don't need to identify which specific tool call was "right".
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, computed_field


class ToolObservation(BaseModel):
    """
    Result/observation from executing a tool call.
    
    Captures stdout, stderr, structured output, and error information.
    """
    stdout: str | None = Field(default=None, description="Standard output from tool")
    stderr: str | None = Field(default=None, description="Standard error from tool")
    result: Any = Field(default=None, description="Structured result data")
    error: str | None = Field(default=None, description="Error message if tool failed")
    exit_code: int | None = Field(default=None, description="Exit code if applicable")
    
    @property
    def success(self) -> bool:
        """Whether the tool execution succeeded."""
        return self.error is None and (self.exit_code is None or self.exit_code == 0)


class WorkflowStep(BaseModel):
    """
    A single step in a workflow trajectory.
    
    Each step captures:
    - The tool that was called
    - The arguments passed
    - The observation/result returned
    - Timing information
    """
    step_index: int = Field(..., description="0-based index of this step in the workflow")
    tool_name: str = Field(..., description="Name of the tool called")
    tool_arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to tool")
    observation: ToolObservation = Field(default_factory=ToolObservation, description="Result from tool execution")
    
    # Timing
    started_at: datetime | None = Field(default=None, description="When tool execution started")
    ended_at: datetime | None = Field(default=None, description="When tool execution ended")
    
    # Optional model reasoning before this step
    reasoning: str | None = Field(default=None, description="Model's reasoning for this tool call")
    
    @computed_field
    @property
    def duration_ms(self) -> float | None:
        """Duration of this step in milliseconds."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() * 1000
        return None
    
    @property
    def succeeded(self) -> bool:
        """Whether this step succeeded."""
        return self.observation.success


class TerminalOutcome(BaseModel):
    """
    Terminal outcome for a complete workflow.
    
    This is the core signal for sequence-level learning:
    - A single success/failure boolean
    - A scalar score (0.0 to 1.0)
    - Explanation for debugging
    
    The score is sparse and terminal - assigned only at the end
    of the entire trajectory, not per-step.
    """
    success: bool = Field(..., description="Whether the workflow succeeded")
    score: float = Field(..., ge=0.0, le=1.0, description="Terminal reward score")
    explanation: str = Field(default="", description="Explanation for the outcome")
    
    # Which verifier produced this outcome
    verifier_name: str = Field(default="", description="Name of verifier that produced this")
    verifier_metadata: dict[str, Any] = Field(default_factory=dict, description="Additional verifier data")
    
    @classmethod
    def passed(cls, explanation: str = "Workflow succeeded", verifier: str = "") -> TerminalOutcome:
        """Create a successful outcome."""
        return cls(success=True, score=1.0, explanation=explanation, verifier_name=verifier)
    
    @classmethod
    def failed(cls, explanation: str = "Workflow failed", verifier: str = "") -> TerminalOutcome:
        """Create a failed outcome."""
        return cls(success=False, score=0.0, explanation=explanation, verifier_name=verifier)
    
    @classmethod
    def partial(cls, score: float, explanation: str = "", verifier: str = "") -> TerminalOutcome:
        """Create a partial success outcome."""
        return cls(success=score >= 0.5, score=score, explanation=explanation, verifier_name=verifier)


class WorkflowMetadata(BaseModel):
    """
    Metadata for a workflow trajectory.
    
    Captures operational metrics that can be used for shaping penalties.
    """
    total_tool_calls: int = Field(default=0, description="Number of tool calls made")
    total_latency_ms: float = Field(default=0.0, description="Total execution time in ms")
    total_tokens: int | None = Field(default=None, description="Total tokens used if available")
    estimated_cost: float | None = Field(default=None, description="Estimated cost in USD")
    
    # Error tracking
    tool_errors: int = Field(default=0, description="Number of tool calls that errored")
    schema_violations: int = Field(default=0, description="Number of schema/usage errors")
    
    # Custom metadata
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class WorkflowTrajectory(BaseModel):
    """
    A complete multi-step workflow trajectory.
    
    This is the primary data structure for recording tool-use episodes:
    
    1. Input: The user's request/prompt
    2. Steps: Ordered list of tool calls with observations
    3. Output: The model's final answer
    4. Outcome: Terminal success/score from verifier
    
    Key design principle: The terminal outcome is a SINGLE sparse reward
    at the end of the trajectory. We do not label individual steps as
    correct/incorrect - the learning signal comes only from the final result.
    
    Example:
        trajectory = WorkflowTrajectory(
            trajectory_id="abc123",
            user_request="Fix the bug in main.py",
            steps=[
                WorkflowStep(step_index=0, tool_name="read_file", ...),
                WorkflowStep(step_index=1, tool_name="edit_file", ...),
                WorkflowStep(step_index=2, tool_name="run_tests", ...),
            ],
            final_answer="I fixed the null pointer exception...",
            outcome=TerminalOutcome(success=True, score=1.0, ...)
        )
    """
    # Identification
    trajectory_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique trajectory ID")
    
    # Input
    system_prompt: str = Field(default="", description="System prompt used")
    user_request: str = Field(..., description="The user's request/prompt")
    
    # Execution trace
    steps: list[WorkflowStep] = Field(default_factory=list, description="Ordered list of tool call steps")
    
    # Output
    final_answer: str | None = Field(default=None, description="Model's final response to user")
    
    # Terminal outcome (THE key signal for sequence-level learning)
    outcome: TerminalOutcome | None = Field(default=None, description="Terminal outcome from verifier")
    
    # Metadata
    metadata: WorkflowMetadata = Field(default_factory=WorkflowMetadata)
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime | None = Field(default=None)
    
    # Tags for filtering/organizing
    tags: dict[str, str] = Field(default_factory=dict)
    
    @computed_field
    @property
    def num_steps(self) -> int:
        """Number of steps in this trajectory."""
        return len(self.steps)
    
    @computed_field
    @property
    def total_duration_ms(self) -> float | None:
        """Total duration of the trajectory in milliseconds."""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() * 1000
        return None
    
    @property
    def is_complete(self) -> bool:
        """Whether the trajectory has a terminal outcome."""
        return self.outcome is not None
    
    @property
    def succeeded(self) -> bool:
        """Whether the trajectory succeeded (requires outcome)."""
        return self.outcome is not None and self.outcome.success
    
    @property
    def terminal_score(self) -> float | None:
        """The terminal reward score (None if not yet evaluated)."""
        return self.outcome.score if self.outcome else None
    
    def add_step(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
        observation: ToolObservation | None = None,
        reasoning: str | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
    ) -> WorkflowStep:
        """Add a new step to the trajectory."""
        step = WorkflowStep(
            step_index=len(self.steps),
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            observation=observation or ToolObservation(),
            reasoning=reasoning,
            started_at=started_at,
            ended_at=ended_at,
        )
        self.steps.append(step)
        
        # Update metadata
        self.metadata.total_tool_calls = len(self.steps)
        if step.duration_ms:
            self.metadata.total_latency_ms += step.duration_ms
        if not step.succeeded:
            self.metadata.tool_errors += 1
        
        return step
    
    def complete(
        self,
        final_answer: str | None = None,
        outcome: TerminalOutcome | None = None,
    ) -> None:
        """Mark the trajectory as complete."""
        self.ended_at = datetime.utcnow()
        if final_answer is not None:
            self.final_answer = final_answer
        if outcome is not None:
            self.outcome = outcome
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowTrajectory:
        """Create from dictionary."""
        return cls.model_validate(data)
    
    def to_jsonl_record(self) -> str:
        """Convert to a single JSONL record."""
        import json
        return json.dumps(self.to_dict(), default=str)


class WorkflowRecorder:
    """
    Records workflow trajectories during execution.
    
    Use this to instrument your agent/rollout code:
    
        recorder = WorkflowRecorder()
        
        with recorder.start_trajectory("Fix the bug") as traj:
            # Tool call 1
            with traj.record_step("read_file", {"path": "main.py"}) as step:
                result = read_file("main.py")
                step.set_result(result)
            
            # Tool call 2
            with traj.record_step("edit_file", {...}) as step:
                result = edit_file(...)
                step.set_result(result)
            
            traj.set_final_answer("Fixed the bug...")
        
        # Later: verify and get terminal score
        trajectory = recorder.trajectories[-1]
    """
    
    def __init__(self):
        self.trajectories: list[WorkflowTrajectory] = []
        self._current_trajectory: WorkflowTrajectory | None = None
    
    def start_trajectory(
        self,
        user_request: str,
        system_prompt: str = "",
        tags: dict[str, str] | None = None,
    ) -> TrajectoryContext:
        """Start recording a new trajectory."""
        trajectory = WorkflowTrajectory(
            user_request=user_request,
            system_prompt=system_prompt,
            tags=tags or {},
        )
        self._current_trajectory = trajectory
        return TrajectoryContext(trajectory, self)
    
    def _finish_trajectory(self, trajectory: WorkflowTrajectory) -> None:
        """Called when a trajectory context exits."""
        if trajectory.ended_at is None:
            trajectory.ended_at = datetime.utcnow()
        self.trajectories.append(trajectory)
        self._current_trajectory = None
    
    @property
    def current(self) -> WorkflowTrajectory | None:
        """The currently recording trajectory."""
        return self._current_trajectory


class TrajectoryContext:
    """Context manager for recording a trajectory."""
    
    def __init__(self, trajectory: WorkflowTrajectory, recorder: WorkflowRecorder):
        self.trajectory = trajectory
        self.recorder = recorder
    
    def __enter__(self) -> TrajectoryContext:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.recorder._finish_trajectory(self.trajectory)
        return False
    
    def record_step(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
        reasoning: str | None = None,
    ) -> StepContext:
        """Record a tool call step."""
        return StepContext(self.trajectory, tool_name, tool_arguments, reasoning)
    
    def set_final_answer(self, answer: str) -> None:
        """Set the final answer."""
        self.trajectory.final_answer = answer
    
    def set_outcome(self, outcome: TerminalOutcome) -> None:
        """Set the terminal outcome."""
        self.trajectory.outcome = outcome


class StepContext:
    """Context manager for recording a single step."""
    
    def __init__(
        self,
        trajectory: WorkflowTrajectory,
        tool_name: str,
        tool_arguments: dict[str, Any],
        reasoning: str | None = None,
    ):
        self.trajectory = trajectory
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments
        self.reasoning = reasoning
        self.started_at: datetime | None = None
        self.observation = ToolObservation()
    
    def __enter__(self) -> StepContext:
        self.started_at = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        ended_at = datetime.utcnow()
        
        if exc_type is not None:
            self.observation.error = str(exc_val)
        
        self.trajectory.add_step(
            tool_name=self.tool_name,
            tool_arguments=self.tool_arguments,
            observation=self.observation,
            reasoning=self.reasoning,
            started_at=self.started_at,
            ended_at=ended_at,
        )
        return False  # Don't suppress exceptions
    
    def set_result(
        self,
        result: Any = None,
        stdout: str | None = None,
        stderr: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        """Set the observation for this step."""
        self.observation.result = result
        self.observation.stdout = stdout
        self.observation.stderr = stderr
        self.observation.exit_code = exit_code
    
    def set_error(self, error: str) -> None:
        """Mark this step as failed."""
        self.observation.error = error
