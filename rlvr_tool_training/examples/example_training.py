"""
Example: Training a Model for Tool Selection

This example demonstrates the complete workflow for training an OSS model
to make better tool call responses using RLVR (Reinforcement Learning from
Verifiable Rewards).

The example shows how to:
1. Specify the model to train
2. Define available tools
3. Create training trajectories with VR signals
4. Run the training pipeline with OpenPipe and W&B
"""

from rlvr_tool_training.config import (
    TrainingConfig,
    ToolRegistry,
    ToolDefinition,
    ToolParameter,
)
from rlvr_tool_training.core import (
    Trajectory,
    ToolCall,
    VerifiableReward,
    TrajectoryGenerator,
    RLVRTrainer,
)
from rlvr_tool_training.config.tool_definitions import create_coding_assistant_tools


def create_sample_trajectories() -> list[Trajectory]:
    """
    Create sample training trajectories with verifiable rewards.
    
    In a real application, these would come from:
    - Logged production data with success/failure outcomes
    - Synthetic data generation with known correct answers
    - Human annotations
    """
    generator = TrajectoryGenerator(
        system_prompt="You are a coding assistant. Use the provided tools to help users with their coding tasks."
    )
    
    trajectories = []
    
    # === EXAMPLE 1: Correct tool selection - reading a file ===
    trajectories.append(generator.create_trajectory(
        user_message="Show me the contents of main.py",
        chosen_tool="read_file",
        chosen_args={"path": "main.py"},
        reward=VerifiableReward.perfect("Correctly chose read_file for viewing file contents"),
    ))
    
    # === EXAMPLE 2: Preference pair - read vs write for viewing ===
    # This creates both a chosen (correct) and rejected (incorrect) response
    trajectories.append(generator.create_preference_pair(
        user_message="What's in the config.json file?",
        good_tool="read_file",
        good_args={"path": "config.json"},
        bad_tool="write_file",  # Wrong: should read, not write
        bad_args={"path": "config.json", "content": ""},
        good_reason="read_file is correct for viewing file contents",
        bad_reason="write_file would overwrite the file instead of reading it"
    ))
    
    # === EXAMPLE 3: Correct tool for searching ===
    trajectories.append(generator.create_trajectory(
        user_message="Find all usages of 'async def' in the project",
        chosen_tool="search_code",
        chosen_args={"pattern": "async def", "directory": "."},
        reward=VerifiableReward.perfect("Correctly chose search_code for pattern search"),
    ))
    
    # === EXAMPLE 4: Partial reward - right tool, imprecise params ===
    trajectories.append(generator.create_trajectory(
        user_message="Search for TODO comments in src/",
        chosen_tool="search_code",
        chosen_args={"pattern": "TODO"},  # Missing directory param
        reward=VerifiableReward.partial(
            "Right tool but missing directory parameter",
            tool_ok=True,
            params_ok=False
        ),
    ))
    
    # === EXAMPLE 5: Wrong tool selection ===
    trajectories.append(generator.create_trajectory(
        user_message="List all Python files in the tests folder",
        chosen_tool="read_file",  # Wrong: should be list_files
        chosen_args={"path": "tests"},
        reward=VerifiableReward.wrong("Should use list_files, not read_file"),
        rejected_tool="search_code",  # Also wrong alternative
        rejected_args={"pattern": "*.py", "directory": "tests"},
    ))
    
    # === EXAMPLE 6: Correct list_files usage ===
    trajectories.append(generator.create_trajectory(
        user_message="What files are in the src directory?",
        chosen_tool="list_files",
        chosen_args={"path": "src"},
        reward=VerifiableReward.perfect("Correctly chose list_files for directory listing"),
    ))
    
    # === EXAMPLE 7: Correct command execution ===
    trajectories.append(generator.create_trajectory(
        user_message="Run the tests",
        chosen_tool="run_command",
        chosen_args={"command": "pytest", "working_dir": "."},
        reward=VerifiableReward.perfect("Correctly chose run_command for test execution"),
    ))
    
    # === EXAMPLE 8: Preference pair for file creation ===
    trajectories.append(generator.create_preference_pair(
        user_message="Create a new file called hello.py with a print statement",
        good_tool="write_file",
        good_args={"path": "hello.py", "content": 'print("Hello, World!")'},
        bad_tool="run_command",
        bad_args={"command": "echo 'print(\"Hello\")' > hello.py"},
        good_reason="write_file is the proper tool for creating files",
        bad_reason="run_command with echo is less safe and portable"
    ))
    
    # === EXAMPLE 9: Complex search scenario ===
    trajectories.append(generator.create_trajectory(
        user_message="Find all error handling code in the api module",
        chosen_tool="search_code",
        chosen_args={"pattern": "except|raise|Error", "directory": "api"},
        reward=VerifiableReward(
            score=0.8,
            reason="Good tool choice, regex could be more precise",
            tool_correct=True,
            params_correct=False
        ),
    ))
    
    # === EXAMPLE 10: Multi-step would need read first ===
    trajectories.append(generator.create_trajectory(
        user_message="Update the version number in setup.py to 2.0.0",
        chosen_tool="read_file",  # Good first step: read before modify
        chosen_args={"path": "setup.py"},
        reward=VerifiableReward(
            score=0.9,
            reason="Correctly reads file first before modification",
            tool_correct=True,
            params_correct=True
        ),
        metadata={"step": "1", "total_steps": "2"}
    ))
    
    return trajectories


def main():
    """
    Main entry point demonstrating the RLVR training workflow.
    
    This shows how to wire together:
    1. Model specification (what to train)
    2. Tool definitions (what tools are available)
    3. VR signals (trajectories with rewards)
    """
    
    print("=" * 60)
    print("RLVR Tool Call Training - Proof of Concept")
    print("=" * 60)
    
    # ============================================================
    # STEP 1: SPECIFY THE MODEL TO TRAIN
    # ============================================================
    print("\n📦 Step 1: Configure the model to train")
    
    # You can specify any supported OSS model
    config = TrainingConfig(
        # The base model to fine-tune
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        
        # Training hyperparameters for RLVR
        hyperparameters={
            "batch_size": "auto",
            "learning_rate_multiplier": 1.0,
            "num_epochs": 3
        },
        
        # OpenPipe settings
        openpipe={
            "dataset_name": "tool-selection-training-v1",
            "model_slug": "tool-selector-llama-8b-v1"
        },
        
        # W&B settings for experiment tracking
        wandb={
            "project": "rlvr-tool-training-poc",
            "tags": ["rlvr", "tool-calling", "llama-8b", "poc"]
        }
    )
    
    print(f"   Model: {config.base_model}")
    print(f"   Epochs: {config.hyperparameters.num_epochs}")
    print(f"   Dataset: {config.openpipe.dataset_name}")
    
    # ============================================================
    # STEP 2: DEFINE AVAILABLE TOOLS
    # ============================================================
    print("\n🔧 Step 2: Define available tools")
    
    # Use the pre-built coding assistant tools
    tools = create_coding_assistant_tools()
    
    # Or define custom tools:
    # tools = ToolRegistry(tools=[
    #     ToolDefinition(
    #         name="my_custom_tool",
    #         description="Does something custom",
    #         parameters=[...]
    #     )
    # ])
    
    print(f"   Available tools ({len(tools.tools)}):")
    for tool in tools.tools:
        print(f"   - {tool.name}: {tool.description[:50]}...")
    
    # ============================================================
    # STEP 3: CREATE TRAJECTORIES WITH VR SIGNALS
    # ============================================================
    print("\n📊 Step 3: Create training trajectories with verifiable rewards")
    
    trajectories = create_sample_trajectories()
    
    print(f"   Total trajectories: {len(trajectories)}")
    
    # Show reward distribution
    rewards = [t.reward.score for t in trajectories]
    print(f"   Reward distribution:")
    print(f"   - Perfect (1.0): {sum(1 for r in rewards if r == 1.0)}")
    print(f"   - Partial (0.5-0.9): {sum(1 for r in rewards if 0.5 <= r < 1.0)}")
    print(f"   - Wrong (0.0): {sum(1 for r in rewards if r == 0.0)}")
    
    # Show preference pairs
    preference_pairs = sum(1 for t in trajectories if t.rejected_tool_call is not None)
    print(f"   Preference pairs: {preference_pairs}")
    
    # ============================================================
    # STEP 4: INITIALIZE TRAINER AND RUN
    # ============================================================
    print("\n🚀 Step 4: Initialize trainer and execute")
    
    # Create the trainer with dry_run=True for demo
    # Set dry_run=False and provide API keys for real training
    trainer = RLVRTrainer(
        config=config,
        tool_registry=tools,
        dry_run=True  # Set to False for actual training
    )
    
    # Add trajectories
    trainer.add_trajectories(trajectories)
    
    # Get training summary before running
    summary = trainer.get_training_summary()
    print(f"\n   Training Summary:")
    print(f"   - Model: {summary['model']}")
    print(f"   - Tools: {summary['tools']}")
    print(f"   - Trajectories: {summary['num_trajectories']}")
    print(f"   - Reward mean: {summary['reward_stats']['mean']:.2f}")
    
    # Execute training (dry run)
    print("\n   Executing training pipeline (dry run)...")
    results = trainer.train()
    
    print(f"\n   ✅ Training pipeline completed!")
    print(f"   - Dataset ID: {results['dataset_id']}")
    print(f"   - Model status: {results['model']['status']}")
    
    # ============================================================
    # STEP 5: REVIEW RESULTS
    # ============================================================
    print("\n📈 Step 5: Training results")
    print(f"   Steps completed:")
    for step in results['steps']:
        print(f"   - {step['name']}: {step['status']}")
    
    print("\n" + "=" * 60)
    print("POC Complete!")
    print("To run actual training:")
    print("1. Set OPENPIPE_API_KEY environment variable")
    print("2. Set WANDB_API_KEY environment variable")
    print("3. Set dry_run=False in RLVRTrainer")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
