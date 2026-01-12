# RLVR Tool Call Training POC

A proof-of-concept demonstrating how to use **Reinforcement Learning from Verifiable Rewards (RLVR)** to train OSS models to make better tool call responses.

This POC integrates:
- **[OpenPipe](https://openpipe.ai/)** - For model fine-tuning with reward-based training
- **[Weights & Biases](https://wandb.ai/)** - For experiment tracking and visualization

## Overview

RLVR training allows you to improve a model's ability to select the right tool and parameters by providing verifiable reward signals. This is particularly useful for:
- Tool-calling agents that need to select from multiple tools
- Code assistants that need to choose between read/write/search operations
- Any system where correct tool selection can be objectively verified

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Example

```bash
python -m rlvr_tool_training.examples.example_training
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Key Concepts

### 1. Specifying the Model to Train

```python
from rlvr_tool_training.config import TrainingConfig

# Specify any supported OSS model
config = TrainingConfig(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    hyperparameters={
        "num_epochs": 3,
        "learning_rate_multiplier": 1.0,
    }
)

# Or use convenience methods
config = TrainingConfig.for_llama_8b()
config = TrainingConfig.for_mistral_7b()
```

### 2. Defining Available Tools

```python
from rlvr_tool_training.config import ToolRegistry, ToolDefinition, ToolParameter

# Define tools the model should learn to select from
tools = ToolRegistry(tools=[
    ToolDefinition(
        name="read_file",
        description="Read the contents of a file",
        parameters=[
            ToolParameter(name="path", type="string", description="File path", required=True)
        ]
    ),
    ToolDefinition(
        name="search_code",
        description="Search for patterns in code",
        parameters=[
            ToolParameter(name="pattern", type="string", description="Regex pattern", required=True),
            ToolParameter(name="directory", type="string", description="Directory to search")
        ]
    ),
])

# Or use pre-built tool sets
from rlvr_tool_training.config.tool_definitions import create_coding_assistant_tools
tools = create_coding_assistant_tools()
```

### 3. Creating Training Trajectories with VR Signals

```python
from rlvr_tool_training.core import TrajectoryGenerator, VerifiableReward

generator = TrajectoryGenerator()

# Create a trajectory with a perfect reward (correct tool + params)
trajectory = generator.create_trajectory(
    user_message="Show me the contents of main.py",
    chosen_tool="read_file",
    chosen_args={"path": "main.py"},
    reward=VerifiableReward.perfect("Correctly chose read_file for viewing file contents")
)

# Create a preference pair (for DPO-style training)
trajectory = generator.create_preference_pair(
    user_message="What's in config.json?",
    good_tool="read_file",
    good_args={"path": "config.json"},
    bad_tool="write_file",  # Wrong choice
    bad_args={"path": "config.json", "content": ""},
)

# Create with partial reward
trajectory = generator.create_trajectory(
    user_message="Search for TODO in src/",
    chosen_tool="search_code",
    chosen_args={"pattern": "TODO"},  # Missing directory
    reward=VerifiableReward.partial("Right tool, incomplete params", tool_ok=True, params_ok=False)
)
```

### 4. Running the Training Pipeline

```python
from rlvr_tool_training.core import RLVRTrainer

trainer = RLVRTrainer(
    config=config,
    tool_registry=tools,
    dry_run=False  # Set True for testing
)

# Add training data
trainer.add_trajectories(trajectories)

# Execute training
results = trainer.train()
print(f"Model ID: {results['model']['model_id']}")
```

## Architecture

```
rlvr_tool_training/
├── config/
│   ├── training_config.py   # Model & hyperparameter configuration
│   └── tool_definitions.py  # Tool schema definitions
├── core/
│   ├── trajectories.py      # Trajectory & reward data structures
│   ├── rewards.py           # Verifiable reward functions
│   └── trainer.py           # OpenPipe + W&B training integration
└── examples/
    └── example_training.py  # Complete usage example
```

## Verifiable Rewards

The VR signals are computed using reward functions:

```python
from rlvr_tool_training.core.rewards import (
    ToolSelectionReward,      # Only checks tool name
    ParameterAccuracyReward,  # Checks tool + parameters
    CompositeReward,          # Combines multiple rewards
    verify_tool_call,         # Convenience function
)

# Simple tool verification
reward = verify_tool_call(tool_call, expected_tool="read_file")

# Strict verification with params
reward = verify_tool_call(
    tool_call,
    expected_tool="search_code",
    expected_args={"pattern": "TODO", "directory": "src"},
    strict=True
)
```

## Configuration Reference

### TrainingConfig

| Field | Type | Description |
|-------|------|-------------|
| `base_model` | str | OSS model to fine-tune (e.g., "meta-llama/Llama-3.1-8B-Instruct") |
| `hyperparameters.num_epochs` | int | Number of training epochs (1-10) |
| `hyperparameters.batch_size` | int\|"auto" | Training batch size |
| `hyperparameters.learning_rate_multiplier` | float | LR multiplier (0.1-10.0) |
| `openpipe.dataset_name` | str | Name for the OpenPipe dataset |
| `openpipe.model_slug` | str | Slug for the fine-tuned model |
| `wandb.project` | str | W&B project name |
| `wandb.tags` | list[str] | Tags for the W&B run |

## Environment Variables

For actual training (not dry_run), set:

```bash
export OPENPIPE_API_KEY="your-openpipe-api-key"
export WANDB_API_KEY="your-wandb-api-key"
```

## Dependencies

- `openpipe==5.0.0` - OpenPipe SDK for fine-tuning
- `wandb==0.23.1` - Weights & Biases for experiment tracking
- `pydantic>=2.0.0` - Data validation
- `pytest>=8.0.0` - Testing

## License

MIT
