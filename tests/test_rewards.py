"""Tests for reward functions."""

import pytest
from rlvr_tool_training.core import ToolCall, VerifiableReward
from rlvr_tool_training.core.rewards import (
    ToolSelectionReward,
    ParameterAccuracyReward,
    CompositeReward,
    verify_tool_call,
)


class TestToolSelectionReward:
    def test_correct_tool(self):
        reward_fn = ToolSelectionReward()
        call = ToolCall(name="read_file", arguments={"path": "test.py"})
        
        reward = reward_fn.compute(call, expected_tool="read_file")
        
        assert reward.score == 1.0
        assert reward.tool_correct is True
    
    def test_wrong_tool(self):
        reward_fn = ToolSelectionReward()
        call = ToolCall(name="write_file", arguments={"path": "test.py"})
        
        reward = reward_fn.compute(call, expected_tool="read_file")
        
        assert reward.score == 0.0
        assert reward.tool_correct is False


class TestParameterAccuracyReward:
    def test_perfect_match(self):
        reward_fn = ParameterAccuracyReward()
        call = ToolCall(name="read_file", arguments={"path": "test.py"})
        
        reward = reward_fn.compute(
            call, 
            expected_tool="read_file",
            expected_args={"path": "test.py"}
        )
        
        assert reward.score == 1.0
        assert reward.tool_correct is True
        assert reward.params_correct is True
    
    def test_correct_tool_partial_params(self):
        reward_fn = ParameterAccuracyReward()
        call = ToolCall(name="search", arguments={"query": "test"})
        
        reward = reward_fn.compute(
            call,
            expected_tool="search",
            expected_args={"query": "test", "directory": "src"}
        )
        
        # Should get partial credit
        assert 0.5 <= reward.score < 1.0
        assert reward.tool_correct is True
    
    def test_wrong_tool(self):
        reward_fn = ParameterAccuracyReward()
        call = ToolCall(name="write_file", arguments={"path": "test.py"})
        
        reward = reward_fn.compute(
            call,
            expected_tool="read_file",
            expected_args={"path": "test.py"}
        )
        
        assert reward.score < 0.5
        assert reward.tool_correct is False
    
    def test_no_expected_args(self):
        reward_fn = ParameterAccuracyReward()
        call = ToolCall(name="read_file", arguments={"path": "test.py"})
        
        reward = reward_fn.compute(call, expected_tool="read_file")
        
        assert reward.score == 1.0


class TestCompositeReward:
    def test_weighted_combination(self):
        tool_reward = ToolSelectionReward()
        param_reward = ParameterAccuracyReward()
        
        composite = CompositeReward([
            (tool_reward, 0.5),
            (param_reward, 0.5)
        ])
        
        call = ToolCall(name="read_file", arguments={"path": "test.py"})
        
        reward = composite.compute(
            call,
            expected_tool="read_file",
            expected_args={"path": "test.py"}
        )
        
        assert reward.score == 1.0  # Both should be perfect


class TestVerifyToolCall:
    def test_simple_verification(self):
        call = ToolCall(name="read_file", arguments={"path": "test.py"})
        
        reward = verify_tool_call(call, expected_tool="read_file")
        
        assert reward.score == 1.0
    
    def test_strict_verification(self):
        call = ToolCall(name="read_file", arguments={"path": "test.py"})
        
        reward = verify_tool_call(
            call,
            expected_tool="read_file",
            expected_args={"path": "test.py"},
            strict=True
        )
        
        assert reward.score == 1.0
