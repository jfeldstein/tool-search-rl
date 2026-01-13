"""Tests for tool selection verification module."""

import pytest

from rlvr_tool_training.core.tool_selection import (
    ToolSelectionQuery,
    ToolSelectionResponse,
    SelectionVerdict,
    GroundTruthVerifier,
    IntentMatchingVerifier,
    RuleBasedVerifier,
    CompositeSelectionVerifier,
    ToolSelectionExample,
    create_selection_example,
)


class TestSelectionVerdict:
    def test_right_tool(self):
        verdict = SelectionVerdict.right_tool("Good choice")
        assert verdict.correct is True
        assert verdict.score == 1.0
        assert verdict.tool_correct is True
    
    def test_wrong_tool(self):
        verdict = SelectionVerdict.wrong_tool("Bad choice")
        assert verdict.correct is False
        assert verdict.score == 0.0
        assert verdict.tool_correct is False
    
    def test_right_tool_wrong_args(self):
        verdict = SelectionVerdict.right_tool_wrong_args()
        assert verdict.correct is False
        assert verdict.score == 0.5
        assert verdict.tool_correct is True
        assert verdict.args_correct is False


class TestGroundTruthVerifier:
    def test_correct_selection(self):
        verifier = GroundTruthVerifier()
        
        query = ToolSelectionQuery(
            user_message="Show me main.py",
            available_tools=["read_file", "write_file"],
            correct_tool="read_file",
            correct_args={"path": "main.py"},
        )
        response = ToolSelectionResponse(
            selected_tool="read_file",
            selected_args={"path": "main.py"},
        )
        
        verdict = verifier.verify(query, response)
        
        assert verdict.correct is True
        assert verdict.score == 1.0
        assert verdict.tool_correct is True
    
    def test_wrong_tool(self):
        verifier = GroundTruthVerifier()
        
        query = ToolSelectionQuery(
            user_message="Show me main.py",
            available_tools=["read_file", "write_file"],
            correct_tool="read_file",
        )
        response = ToolSelectionResponse(
            selected_tool="write_file",  # Wrong!
            selected_args={"path": "main.py"},
        )
        
        verdict = verifier.verify(query, response)
        
        assert verdict.correct is False
        assert verdict.score == 0.0
        assert verdict.tool_correct is False
    
    def test_right_tool_wrong_args(self):
        verifier = GroundTruthVerifier(check_args=True)
        
        query = ToolSelectionQuery(
            user_message="Search in api/",
            available_tools=["search_code"],
            correct_tool="search_code",
            correct_args={"pattern": "TODO", "directory": "api"},
        )
        response = ToolSelectionResponse(
            selected_tool="search_code",
            selected_args={"pattern": "TODO"},  # Missing directory
        )
        
        verdict = verifier.verify(query, response)
        
        assert verdict.tool_correct is True
        assert verdict.args_correct is False
        assert verdict.score == 0.5
    
    def test_no_ground_truth(self):
        verifier = GroundTruthVerifier()
        
        query = ToolSelectionQuery(
            user_message="Do something",
            available_tools=["tool1"],
            correct_tool=None,  # No ground truth
        )
        response = ToolSelectionResponse(selected_tool="tool1", selected_args={})
        
        verdict = verifier.verify(query, response)
        
        assert verdict.score == 0.5  # Inconclusive
    
    def test_fuzzy_args_matching(self):
        verifier = GroundTruthVerifier(fuzzy_args=True)
        
        query = ToolSelectionQuery(
            user_message="Read file",
            available_tools=["read_file"],
            correct_tool="read_file",
            correct_args={"path": "main.py"},
        )
        response = ToolSelectionResponse(
            selected_tool="read_file",
            selected_args={"path": "src/main.py"},  # Contains "main.py"
        )
        
        verdict = verifier.verify(query, response)
        
        assert verdict.tool_correct is True


class TestIntentMatchingVerifier:
    def test_read_intent(self):
        verifier = IntentMatchingVerifier()
        
        query = ToolSelectionQuery(
            user_message="Show me the contents of config.json",
            available_tools=["read_file", "write_file", "search_code"],
        )
        
        # Correct tool for read intent
        response = ToolSelectionResponse(selected_tool="read_file", selected_args={})
        verdict = verifier.verify(query, response)
        assert verdict.correct is True
        
        # Wrong tool for read intent
        response = ToolSelectionResponse(selected_tool="write_file", selected_args={})
        verdict = verifier.verify(query, response)
        assert verdict.correct is False
    
    def test_search_intent(self):
        verifier = IntentMatchingVerifier()
        
        query = ToolSelectionQuery(
            user_message="Find all TODO comments in the codebase",
            available_tools=["read_file", "search_code"],
        )
        
        response = ToolSelectionResponse(selected_tool="search_code", selected_args={})
        verdict = verifier.verify(query, response)
        
        assert verdict.correct is True
    
    def test_list_intent(self):
        verifier = IntentMatchingVerifier()
        
        query = ToolSelectionQuery(
            user_message="What files are in the src directory?",
            available_tools=["list_files", "read_file"],
        )
        
        response = ToolSelectionResponse(selected_tool="list_files", selected_args={})
        verdict = verifier.verify(query, response)
        
        assert verdict.correct is True
    
    def test_run_intent(self):
        verifier = IntentMatchingVerifier()
        
        query = ToolSelectionQuery(
            user_message="Run the pytest tests",
            available_tools=["run_command", "read_file"],
        )
        
        response = ToolSelectionResponse(selected_tool="run_command", selected_args={})
        verdict = verifier.verify(query, response)
        
        assert verdict.correct is True
    
    def test_no_clear_intent(self):
        verifier = IntentMatchingVerifier()
        
        query = ToolSelectionQuery(
            user_message="Do something interesting",
            available_tools=["tool1"],
        )
        
        response = ToolSelectionResponse(selected_tool="tool1", selected_args={})
        verdict = verifier.verify(query, response)
        
        # Should give benefit of doubt
        assert verdict.score == 0.5


class TestRuleBasedVerifier:
    def test_custom_rule(self):
        verifier = RuleBasedVerifier()
        
        # Add a rule: "read_file" is wrong if message contains "create"
        def no_read_for_create(query, response):
            if "create" in query.user_message.lower() and response.selected_tool == "read_file":
                return SelectionVerdict.wrong_tool("Can't read a file you're creating")
            return None
        
        verifier.add_rule(no_read_for_create)
        
        query = ToolSelectionQuery(
            user_message="Create a new file",
            available_tools=["read_file", "write_file"],
        )
        
        response = ToolSelectionResponse(selected_tool="read_file", selected_args={})
        verdict = verifier.verify(query, response)
        
        assert verdict.correct is False
    
    def test_no_matching_rule(self):
        verifier = RuleBasedVerifier()
        
        query = ToolSelectionQuery(
            user_message="Do something",
            available_tools=["tool1"],
        )
        response = ToolSelectionResponse(selected_tool="tool1", selected_args={})
        
        verdict = verifier.verify(query, response)
        
        assert verdict.score == 0.5  # Inconclusive


class TestCompositeSelectionVerifier:
    def test_priority_ordering(self):
        # Ground truth should take priority
        gt_verifier = GroundTruthVerifier()
        intent_verifier = IntentMatchingVerifier()
        
        composite = CompositeSelectionVerifier([gt_verifier, intent_verifier])
        
        query = ToolSelectionQuery(
            user_message="Show me the file",
            available_tools=["read_file", "write_file"],
            correct_tool="write_file",  # Ground truth says write (unusual)
        )
        response = ToolSelectionResponse(selected_tool="write_file", selected_args={})
        
        verdict = composite.verify(query, response)
        
        # Should use ground truth (write_file is correct per GT)
        assert verdict.correct is True


class TestToolSelectionExample:
    def test_create_example(self):
        example = create_selection_example(
            user_message="Read main.py",
            available_tools=["read_file", "write_file"],
            selected_tool="read_file",
            selected_args={"path": "main.py"},
            correct_tool="read_file",
            correct_args={"path": "main.py"},
        )
        
        assert example.query.user_message == "Read main.py"
        assert example.response.selected_tool == "read_file"
        assert example.verdict.correct is True
        assert example.verdict.score == 1.0
    
    def test_example_with_rejected(self):
        example = create_selection_example(
            user_message="Read config.json",
            available_tools=["read_file", "write_file"],
            selected_tool="read_file",
            selected_args={"path": "config.json"},
            correct_tool="read_file",
            rejected_tool="write_file",
            rejected_args={"path": "config.json"},
        )
        
        assert example.rejected_response is not None
        assert example.rejected_response.selected_tool == "write_file"
    
    def test_to_training_entry(self):
        example = create_selection_example(
            user_message="Show me main.py",
            available_tools=["read_file"],
            selected_tool="read_file",
            selected_args={"path": "main.py"},
            correct_tool="read_file",
        )
        
        tool_schemas = [{"type": "function", "function": {"name": "read_file"}}]
        entry = example.to_training_entry(tool_schemas)
        
        assert "messages" in entry
        assert "tools" in entry
        assert "metadata" in entry
        assert entry["metadata"]["reward_score"] == "1.0"
        assert entry["metadata"]["tool_correct"] == "True"
