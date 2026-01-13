"""Tests for trajectory and reward modules."""

import pytest
from rlvr_tool_training.core import (
    Trajectory,
    ToolCall,
    VerifiableReward,
    TrajectoryGenerator,
)


class TestVerifiableReward:
    def test_perfect_reward(self):
        reward = VerifiableReward.perfect("Great job")
        
        assert reward.score == 1.0
        assert reward.tool_correct is True
        assert reward.params_correct is True
    
    def test_partial_reward(self):
        reward = VerifiableReward.partial("Good but not perfect", tool_ok=True)
        
        assert reward.score == 0.5
        assert reward.tool_correct is True
        assert reward.params_correct is False
    
    def test_wrong_reward(self):
        reward = VerifiableReward.wrong("Incorrect")
        
        assert reward.score == 0.0
        assert reward.tool_correct is False
        assert reward.params_correct is False
    
    def test_custom_reward(self):
        reward = VerifiableReward(
            score=0.75,
            reason="Mostly correct",
            tool_correct=True,
            params_correct=False
        )
        
        assert reward.score == 0.75


class TestToolCall:
    def test_basic_tool_call(self):
        call = ToolCall(name="read_file", arguments={"path": "test.py"})
        
        assert call.name == "read_file"
        assert call.arguments["path"] == "test.py"
    
    def test_to_openai_format(self):
        call = ToolCall(name="search", arguments={"query": "test"})
        
        openai_format = call.to_openai_format()
        
        assert "id" in openai_format
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "search"
        assert "arguments" in openai_format["function"]


class TestTrajectory:
    def test_basic_trajectory(self):
        trajectory = Trajectory(
            user_message="Read the file",
            tool_call=ToolCall(name="read_file", arguments={"path": "test.py"}),
            reward=VerifiableReward.perfect()
        )
        
        assert trajectory.user_message == "Read the file"
        assert trajectory.tool_call.name == "read_file"
        assert trajectory.reward.score == 1.0
    
    def test_trajectory_with_rejected(self):
        trajectory = Trajectory(
            user_message="Read the file",
            tool_call=ToolCall(name="read_file", arguments={"path": "test.py"}),
            rejected_tool_call=ToolCall(name="write_file", arguments={"path": "test.py", "content": ""}),
            reward=VerifiableReward.perfect()
        )
        
        assert trajectory.rejected_tool_call is not None
        assert trajectory.rejected_tool_call.name == "write_file"
    
    def test_to_openpipe_entry(self):
        trajectory = Trajectory(
            user_message="Read the file",
            tool_call=ToolCall(name="read_file", arguments={"path": "test.py"}),
            reward=VerifiableReward.perfect("Correct")
        )
        
        tools = [{"type": "function", "function": {"name": "read_file", "description": "Read", "parameters": {}}}]
        entry = trajectory.to_openpipe_entry(tools)
        
        assert "messages" in entry
        assert "tools" in entry
        assert len(entry["messages"]) == 3  # system, user, assistant
        assert entry["messages"][1]["content"] == "Read the file"
        assert entry["metadata"]["reward_score"] == "1.0"


class TestTrajectoryGenerator:
    def test_create_trajectory(self):
        generator = TrajectoryGenerator()
        
        trajectory = generator.create_trajectory(
            user_message="Test message",
            chosen_tool="test_tool",
            chosen_args={"arg": "value"},
            reward=VerifiableReward.perfect()
        )
        
        assert trajectory.user_message == "Test message"
        assert trajectory.tool_call.name == "test_tool"
        assert trajectory.tool_call.arguments["arg"] == "value"
    
    def test_create_preference_pair(self):
        generator = TrajectoryGenerator()
        
        trajectory = generator.create_preference_pair(
            user_message="Test",
            good_tool="good",
            good_args={"x": 1},
            bad_tool="bad",
            bad_args={"x": 2}
        )
        
        assert trajectory.tool_call.name == "good"
        assert trajectory.rejected_tool_call is not None
        assert trajectory.rejected_tool_call.name == "bad"
        assert trajectory.reward.score == 1.0
    
    def test_custom_system_prompt(self):
        generator = TrajectoryGenerator(system_prompt="Custom prompt")
        
        trajectory = generator.create_trajectory(
            user_message="Test",
            chosen_tool="tool",
            chosen_args={},
            reward=VerifiableReward.perfect()
        )
        
        assert trajectory.system_prompt == "Custom prompt"
