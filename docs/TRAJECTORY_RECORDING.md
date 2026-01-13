# Trajectory Recording and Terminal Verification

This document describes how to record complete tool-use trajectories and compute terminal rewards for sequence-level learning.

## Overview

The trajectory recording system enables:
- Recording complete tool-use workflows (user request → tool calls → observations → final answer)
- Computing terminal "workflow success" scores using pluggable verifiers
- Sequence-level learning where the reward signal is sparse and terminal

**Key Principle**: The core signal is a SINGLE terminal reward assigned at the end of the trajectory. We do NOT label individual tool calls as correct/incorrect.

## Quick Start

### Recording Trajectories

```python
from rlvr_tool_training.core import (
    RolloutManager,
    RolloutConfig,
    WorkflowCompletionVerifier,
)

# Configure the rollout manager
config = RolloutConfig(
    output_dir="./trajectories",
    auto_export=True,
    verifier=WorkflowCompletionVerifier(max_steps=20),
)
manager = RolloutManager(config)

# Record a trajectory
with manager.start_rollout("Fix the bug in main.py") as rollout:
    # Record tool calls as they happen
    rollout.record_tool_call(
        tool_name="read_file",
        tool_arguments={"path": "main.py"},
        result="file contents...",
    )
    
    rollout.record_tool_call(
        tool_name="edit_file",
        tool_arguments={"path": "main.py", "content": "fixed code"},
        result="File saved",
    )
    
    rollout.record_tool_call(
        tool_name="run_tests",
        tool_arguments={"command": "pytest"},
        stdout="3 passed",
        exit_code=0,
    )
    
    rollout.set_final_answer("Fixed the null pointer exception")

# Check result
print(f"Success: {manager.last_result.success}")
print(f"Score: {manager.last_result.final_score}")
```

### Using Verifiers

```python
from rlvr_tool_training.core import (
    PytestVerifier,
    ArtifactVerifier,
    CompositeVerifier,
)

# Pytest verifier - runs tests and maps results to score
pytest_verifier = PytestVerifier(
    test_path="tests/",
    timeout=300,
)

# Artifact verifier - checks if files were created
artifact_verifier = ArtifactVerifier(
    required_files=["output.json"],
    schema_validators={"output.json": {"type": "object", "required": ["status"]}},
)

# Combine verifiers
verifier = CompositeVerifier(
    verifiers=[pytest_verifier, artifact_verifier],
    mode="all",  # All must pass
)
```

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TRAJECTORY LIFECYCLE                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. RECORDING                    2. VERIFICATION                        │
│  ┌─────────────────────┐         ┌─────────────────────┐               │
│  │ User Request        │         │ Terminal Verifier   │               │
│  │        ↓            │         │   - PytestVerifier  │               │
│  │ Tool Call 1         │   ───►  │   - ArtifactVerifier│               │
│  │ Tool Call 2         │         │   - Custom Verifier │               │
│  │ Tool Call N         │         └─────────┬───────────┘               │
│  │        ↓            │                   │                            │
│  │ Final Answer        │                   ▼                            │
│  └─────────────────────┘         ┌─────────────────────┐               │
│                                  │ Terminal Score      │               │
│  3. SHAPING PENALTIES            │ (0.0 to 1.0)        │               │
│  ┌─────────────────────┐         └─────────────────────┘               │
│  │ - Too many calls    │                   │                            │
│  │ - Excessive latency │   ───────────────►│                            │
│  │ - Tool errors       │                   ▼                            │
│  └─────────────────────┘         ┌─────────────────────┐               │
│                                  │ Final Reward        │               │
│                                  │ (sparse, terminal)  │               │
│                                  └─────────────────────┘               │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Data Model

### WorkflowTrajectory

The main data structure for a complete episode:

```python
@dataclass
class WorkflowTrajectory:
    trajectory_id: str              # Unique ID
    user_request: str               # The user's request
    system_prompt: str              # System prompt used
    steps: list[WorkflowStep]       # Ordered tool call steps
    final_answer: str | None        # Model's final response
    outcome: TerminalOutcome | None # Terminal score (THE key field)
    metadata: WorkflowMetadata      # Operational metrics
```

### WorkflowStep

A single tool call step:

```python
@dataclass
class WorkflowStep:
    step_index: int                 # Position in trajectory
    tool_name: str                  # Tool that was called
    tool_arguments: dict            # Arguments passed
    observation: ToolObservation    # Result from tool
    started_at: datetime | None     # Timing
    ended_at: datetime | None
    reasoning: str | None           # Model's reasoning
```

### TerminalOutcome

The terminal reward signal:

```python
@dataclass
class TerminalOutcome:
    success: bool                   # Pass/fail
    score: float                    # 0.0 to 1.0
    explanation: str                # For debugging
    verifier_name: str              # Which verifier produced this
```

## Terminal Verifiers

### Built-in Verifiers

| Verifier | Description | Use Case |
|----------|-------------|----------|
| `WorkflowCompletionVerifier` | Checks basic completion | Default/baseline |
| `PytestVerifier` | Runs pytest | Code tasks |
| `ArtifactVerifier` | Checks file artifacts | File generation |
| `FunctionVerifier` | Custom function | Quick custom checks |
| `CompositeVerifier` | Combines verifiers | Complex validation |

### Creating a Custom Verifier

```python
from rlvr_tool_training.core import TerminalVerifier, TerminalOutcome

class MyCustomVerifier(TerminalVerifier):
    @property
    def name(self) -> str:
        return "my_verifier"
    
    def verify(self, trajectory: WorkflowTrajectory) -> TerminalOutcome:
        # Your verification logic
        if some_condition(trajectory):
            return TerminalOutcome.passed("All checks passed", self.name)
        return TerminalOutcome.failed("Check failed", self.name)
```

## Shaping Penalties

Optional trajectory-level penalties (NOT per-step labels):

```python
from rlvr_tool_training.core import (
    TerminalRewardConfig,
    too_many_calls_penalty,
    excessive_latency_penalty,
    tool_error_penalty,
)

config = TerminalRewardConfig(
    penalties=[
        too_many_calls_penalty(max_calls=10, penalty_per_extra=0.05),
        excessive_latency_penalty(max_latency_ms=60000),
        tool_error_penalty(penalty_per_error=0.1),
    ]
)
```

## Export and Statistics

### JSONL Export

```python
from rlvr_tool_training.core import export_trajectories_jsonl

# Export to JSONL (one record per line)
export_trajectories_jsonl(trajectories, "data/trajectories.jsonl")
```

Each record contains:
- Full trajectory trace (all steps)
- Terminal reward/score
- Metadata

### Statistics

```python
from rlvr_tool_training.core import compute_dataset_stats

stats = compute_dataset_stats(trajectories)
print(f"Success rate: {stats.success_rate:.1%}")
print(f"Avg steps: {stats.avg_steps:.1f}")
print(f"Avg latency: {stats.avg_latency_ms:.0f}ms")
```

## Integration with Training

### Converting to Training Format

```python
from rlvr_tool_training.core import load_trajectories_jsonl

# Load trajectories
trajectories = load_trajectories_jsonl("data/trajectories.jsonl")

# Filter by outcome
successful = [t for t in trajectories if t.succeeded]
failed = [t for t in trajectories if not t.succeeded]

# Use for preference learning
# successful trajectories = "chosen"
# failed trajectories = "rejected"
```

## Output Locations

By default, trajectories are written to:
- `./trajectories/trajectories_YYYYMMDD.jsonl` (auto-export)
- Custom path via `export_all(filename="custom.jsonl")`

## Configuration Options

```python
RolloutConfig(
    output_dir="./trajectories",     # Where to write files
    auto_export=True,                # Export after each trajectory
    export_format="jsonl",           # "jsonl" or "json"
    verifier=...,                    # Terminal verifier to use
    reward_config=...,               # Shaping penalty config
    max_steps=50,                    # Maximum steps per trajectory
    enable_logging=True,             # Print progress
)
```
