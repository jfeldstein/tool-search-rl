"""
Tool Selection RLVR Training Example

This example demonstrates the core POC:
Training a model to select the RIGHT tool at the RIGHT time.

Key concepts:
1. Tool selection is VERIFIABLE - we can objectively check if the choice was correct
2. We don't need to execute tools - just verify the selection decision
3. RLVR uses these verifiable signals to improve tool selection

The flow:
    User Query → Model Selects Tool → Verify Selection → Reward Signal → Training
"""

from rlvr_tool_training.config import (
    TrainingConfig,
    ToolRegistry,
    ToolDefinition,
    ToolParameter,
)
from rlvr_tool_training.core.tool_selection import (
    ToolSelectionQuery,
    ToolSelectionResponse,
    SelectionVerdict,
    GroundTruthVerifier,
    IntentMatchingVerifier,
    CompositeSelectionVerifier,
    ToolSelectionExample,
    create_selection_example,
)
from rlvr_tool_training.core import RLVRTrainer


def create_tool_selection_dataset() -> list[ToolSelectionExample]:
    """
    Create a dataset of tool selection examples with verifiable rewards.
    
    Each example has:
    - A user query
    - The model's tool selection (chosen)
    - A verification verdict (the reward signal)
    - Optionally: a rejected alternative (for preference learning)
    """
    
    available_tools = ["read_file", "write_file", "search_code", "run_command", "list_files"]
    
    examples = []
    
    # === CORRECT SELECTIONS (reward = 1.0) ===
    
    # Reading files
    examples.append(create_selection_example(
        user_message="Show me the contents of main.py",
        available_tools=available_tools,
        selected_tool="read_file",
        selected_args={"path": "main.py"},
        correct_tool="read_file",
        correct_args={"path": "main.py"},
    ))
    
    examples.append(create_selection_example(
        user_message="What's in the config.json file?",
        available_tools=available_tools,
        selected_tool="read_file",
        selected_args={"path": "config.json"},
        correct_tool="read_file",
        correct_args={"path": "config.json"},
    ))
    
    # Searching code
    examples.append(create_selection_example(
        user_message="Find all TODO comments in the codebase",
        available_tools=available_tools,
        selected_tool="search_code",
        selected_args={"pattern": "TODO"},
        correct_tool="search_code",
        correct_args={"pattern": "TODO"},
    ))
    
    examples.append(create_selection_example(
        user_message="Where is the function calculate_total defined?",
        available_tools=available_tools,
        selected_tool="search_code",
        selected_args={"pattern": "def calculate_total"},
        correct_tool="search_code",
        correct_args={"pattern": "calculate_total"},
    ))
    
    # Listing files
    examples.append(create_selection_example(
        user_message="What files are in the src directory?",
        available_tools=available_tools,
        selected_tool="list_files",
        selected_args={"path": "src"},
        correct_tool="list_files",
        correct_args={"path": "src"},
    ))
    
    # Running commands
    examples.append(create_selection_example(
        user_message="Run the test suite",
        available_tools=available_tools,
        selected_tool="run_command",
        selected_args={"command": "pytest"},
        correct_tool="run_command",
        correct_args={"command": "pytest"},
    ))
    
    # Writing files
    examples.append(create_selection_example(
        user_message="Create a new file called hello.py with a hello world program",
        available_tools=available_tools,
        selected_tool="write_file",
        selected_args={"path": "hello.py", "content": "print('Hello, World!')"},
        correct_tool="write_file",
    ))
    
    # === INCORRECT SELECTIONS WITH PREFERENCE PAIRS (for DPO-style training) ===
    
    # Wrong: Using write_file when should read
    examples.append(create_selection_example(
        user_message="Show me what's in setup.py",
        available_tools=available_tools,
        selected_tool="read_file",  # Correct choice
        selected_args={"path": "setup.py"},
        correct_tool="read_file",
        rejected_tool="write_file",  # Wrong choice
        rejected_args={"path": "setup.py", "content": ""},
    ))
    
    # Wrong: Using read_file when should list
    examples.append(create_selection_example(
        user_message="List all Python files in tests/",
        available_tools=available_tools,
        selected_tool="list_files",  # Correct choice
        selected_args={"path": "tests"},
        correct_tool="list_files",
        rejected_tool="read_file",  # Wrong choice
        rejected_args={"path": "tests"},
    ))
    
    # Wrong: Using run_command when should search
    examples.append(create_selection_example(
        user_message="Find all usages of 'async def' in the project",
        available_tools=available_tools,
        selected_tool="search_code",  # Correct choice
        selected_args={"pattern": "async def"},
        correct_tool="search_code",
        rejected_tool="run_command",  # Wrong choice
        rejected_args={"command": "grep 'async def'"},
    ))
    
    # === PARTIALLY CORRECT (reward = 0.5) - right tool, wrong args ===
    
    examples.append(create_selection_example(
        user_message="Search for error handling in the api/ directory",
        available_tools=available_tools,
        selected_tool="search_code",  # Right tool
        selected_args={"pattern": "error"},  # Missing directory
        correct_tool="search_code",
        correct_args={"pattern": "error", "directory": "api"},
    ))
    
    return examples


def verify_selections_with_intent_matching():
    """
    Demonstrate verification using intent matching (no ground truth needed).
    
    This is useful when you don't have labeled data - the verifier
    infers correctness from the user's intent.
    """
    print("\n" + "=" * 60)
    print("INTENT-BASED VERIFICATION (no ground truth needed)")
    print("=" * 60)
    
    verifier = IntentMatchingVerifier()
    
    test_cases = [
        # (user_message, selected_tool, expected_correct)
        ("Show me the contents of main.py", "read_file", True),
        ("Show me the contents of main.py", "write_file", False),
        ("Find all TODO comments", "search_code", True),
        ("Find all TODO comments", "read_file", False),
        ("What files are in src/?", "list_files", True),
        ("What files are in src/?", "search_code", False),
        ("Run the tests", "run_command", True),
        ("Run the tests", "read_file", False),
    ]
    
    for user_msg, tool, expected in test_cases:
        query = ToolSelectionQuery(
            user_message=user_msg,
            available_tools=["read_file", "write_file", "search_code", "list_files", "run_command"],
        )
        response = ToolSelectionResponse(selected_tool=tool, selected_args={})
        
        verdict = verifier.verify(query, response)
        
        status = "✓" if verdict.correct == expected else "✗"
        print(f"{status} '{user_msg[:40]}...' → {tool}: score={verdict.score:.1f} ({verdict.reason[:50]})")


def main():
    """
    Main example: Train a model to select the right tool.
    """
    print("=" * 60)
    print("TOOL SELECTION RLVR TRAINING")
    print("=" * 60)
    print("\nGoal: Train model to select the RIGHT tool at the RIGHT time")
    print("Method: Verifiable rewards based on selection correctness\n")
    
    # === STEP 1: Define available tools ===
    print("📋 Step 1: Define available tools")
    
    tools = ToolRegistry(tools=[
        ToolDefinition(
            name="read_file",
            description="Read the contents of a file at the specified path",
            parameters=[ToolParameter(name="path", type="string", description="File path", required=True)]
        ),
        ToolDefinition(
            name="write_file", 
            description="Write content to a file, creating it if needed",
            parameters=[
                ToolParameter(name="path", type="string", description="File path", required=True),
                ToolParameter(name="content", type="string", description="Content to write", required=True),
            ]
        ),
        ToolDefinition(
            name="search_code",
            description="Search for patterns in code files using regex",
            parameters=[
                ToolParameter(name="pattern", type="string", description="Search pattern", required=True),
                ToolParameter(name="directory", type="string", description="Directory to search"),
            ]
        ),
        ToolDefinition(
            name="list_files",
            description="List files and directories at a path",
            parameters=[ToolParameter(name="path", type="string", description="Directory path", required=True)]
        ),
        ToolDefinition(
            name="run_command",
            description="Execute a shell command",
            parameters=[ToolParameter(name="command", type="string", description="Command to run", required=True)]
        ),
    ])
    
    print(f"   Tools: {[t.name for t in tools.tools]}")
    
    # === STEP 2: Create training data with verifiable rewards ===
    print("\n📊 Step 2: Create training data with verifiable rewards")
    
    selection_examples = create_tool_selection_dataset()
    
    # Show statistics
    correct = sum(1 for e in selection_examples if e.verdict.correct)
    with_preferences = sum(1 for e in selection_examples if e.rejected_response)
    
    print(f"   Total examples: {len(selection_examples)}")
    print(f"   Correct selections: {correct}")
    print(f"   Preference pairs: {with_preferences}")
    
    # Show a few examples
    print("\n   Sample examples:")
    for i, ex in enumerate(selection_examples[:3]):
        print(f"   {i+1}. '{ex.query.user_message[:40]}...'")
        print(f"      Selected: {ex.response.selected_tool} → Score: {ex.verdict.score}")
    
    # === STEP 3: Verify the rewards are correct ===
    print("\n🔍 Step 3: Verify reward signals")
    
    verifier = GroundTruthVerifier()
    
    for ex in selection_examples[:5]:
        verdict = verifier.verify(ex.query, ex.response)
        status = "✓" if verdict.correct else "✗"
        print(f"   {status} {ex.response.selected_tool}: {verdict.reason[:50]}")
    
    # === STEP 4: Configure training ===
    print("\n⚙️ Step 4: Configure RLVR training")
    
    config = TrainingConfig(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        hyperparameters={"num_epochs": 3},
        openpipe={
            "dataset_name": "tool-selection-rlvr",
            "model_slug": "tool-selector-v1",
        },
        wandb={
            "project": "tool-selection-rlvr",
            "tags": ["rlvr", "tool-selection"],
        }
    )
    
    print(f"   Model: {config.base_model}")
    print(f"   Dataset: {config.openpipe.dataset_name}")
    
    # === STEP 5: Convert to training format ===
    print("\n📦 Step 5: Convert to OpenPipe training format")
    
    tool_schemas = tools.to_openai_tools()
    training_entries = [ex.to_training_entry(tool_schemas) for ex in selection_examples]
    
    print(f"   Generated {len(training_entries)} training entries")
    
    # Show sample entry structure
    sample = training_entries[0]
    print(f"   Entry structure:")
    print(f"     - messages: {len(sample['messages'])} messages")
    print(f"     - tools: {len(sample['tools'])} tool definitions")
    print(f"     - metadata.reward_score: {sample['metadata']['reward_score']}")
    print(f"     - metadata.tool_correct: {sample['metadata']['tool_correct']}")
    
    # === STEP 6: Show what would be sent to OpenPipe ===
    print("\n🚀 Step 6: Training pipeline (dry run)")
    print("   Would call OpenPipe APIs:")
    print(f"   1. create_dataset(name='{config.openpipe.dataset_name}')")
    print(f"   2. create_dataset_entries(entries=[{len(training_entries)} entries])")
    print(f"   3. create_model(")
    print(f"        slug='{config.openpipe.model_slug}',")
    print(f"        training_config={{")
    print(f"          'provider': 'openpipeReward',")
    print(f"          'base_model': '{config.base_model}'")
    print(f"        }}")
    print(f"      )")
    
    print("\n" + "=" * 60)
    print("SUMMARY: Tool Selection RLVR")
    print("=" * 60)
    print("""
    This POC demonstrates:
    
    1. VERIFIABLE REWARDS for tool selection
       - Compare against ground truth
       - Or use intent matching (no labels needed)
    
    2. NO TOOL EXECUTION required
       - We verify the SELECTION decision
       - Not whether the tool succeeded
    
    3. PREFERENCE LEARNING support
       - Pairs of (good_choice, bad_choice)
       - For DPO-style training
    
    4. SPARSE TERMINAL REWARD
       - One score per selection
       - 1.0 = right tool, 0.5 = right tool/wrong args, 0.0 = wrong tool
    """)
    
    # Also show intent-based verification
    verify_selections_with_intent_matching()
    
    return selection_examples


if __name__ == "__main__":
    main()
