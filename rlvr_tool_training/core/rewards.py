"""
Reward Functions Module

Defines verifiable reward functions for evaluating tool call quality.
These are the "VR signals" in RLVR - they provide objective feedback
on whether a tool call was correct.
"""

from abc import ABC, abstractmethod
from typing import Any
from .trajectories import ToolCall, VerifiableReward


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    
    Reward functions compute a verifiable reward signal for a given
    tool call. They should be deterministic and objectively computable.
    """
    
    @abstractmethod
    def compute(
        self, 
        tool_call: ToolCall, 
        expected_tool: str,
        expected_args: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None
    ) -> VerifiableReward:
        """
        Compute the reward for a tool call.
        
        Args:
            tool_call: The tool call made by the model
            expected_tool: The expected/correct tool name
            expected_args: The expected/correct arguments (optional)
            context: Additional context for reward computation
            
        Returns:
            A VerifiableReward object
        """
        pass


class ToolSelectionReward(RewardFunction):
    """
    Simple reward function that only checks if the correct tool was selected.
    
    Returns:
    - 1.0 if the correct tool was selected
    - 0.0 if the wrong tool was selected
    """
    
    def compute(
        self,
        tool_call: ToolCall,
        expected_tool: str,
        expected_args: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None
    ) -> VerifiableReward:
        if tool_call.name == expected_tool:
            return VerifiableReward(
                score=1.0,
                reason=f"Correctly selected tool: {expected_tool}",
                tool_correct=True,
                params_correct=True  # We're not checking params in this reward
            )
        else:
            return VerifiableReward(
                score=0.0,
                reason=f"Selected {tool_call.name}, expected {expected_tool}",
                tool_correct=False,
                params_correct=False
            )


class ParameterAccuracyReward(RewardFunction):
    """
    Reward function that checks both tool selection and parameter accuracy.
    
    Returns:
    - 1.0 if tool and all required parameters are correct
    - 0.5 if tool is correct but parameters are partially wrong
    - 0.25 if tool is wrong but parameters match the expected tool's schema
    - 0.0 if completely wrong
    """
    
    def __init__(self, required_params_weight: float = 0.7):
        """
        Args:
            required_params_weight: How much weight to give to required params vs optional
        """
        self.required_params_weight = required_params_weight
    
    def compute(
        self,
        tool_call: ToolCall,
        expected_tool: str,
        expected_args: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None
    ) -> VerifiableReward:
        tool_correct = tool_call.name == expected_tool
        
        if expected_args is None:
            # If no expected args provided, just check tool name
            if tool_correct:
                return VerifiableReward.perfect(f"Correct tool: {expected_tool}")
            else:
                return VerifiableReward.wrong(f"Expected {expected_tool}, got {tool_call.name}")
        
        # Check parameter accuracy
        params_score = self._compute_param_score(tool_call.arguments, expected_args)
        params_correct = params_score >= 0.9
        
        if tool_correct and params_correct:
            return VerifiableReward(
                score=1.0,
                reason="Perfect tool call with correct parameters",
                tool_correct=True,
                params_correct=True
            )
        elif tool_correct:
            # Right tool, partially correct params
            score = 0.5 + (params_score * 0.4)  # Score between 0.5 and 0.9
            return VerifiableReward(
                score=score,
                reason=f"Correct tool, parameter accuracy: {params_score:.0%}",
                tool_correct=True,
                params_correct=False
            )
        elif params_score > 0.5:
            # Wrong tool but reasonable params
            return VerifiableReward(
                score=0.25,
                reason=f"Wrong tool ({tool_call.name}), but params reasonable",
                tool_correct=False,
                params_correct=False
            )
        else:
            return VerifiableReward.wrong(
                f"Wrong tool ({tool_call.name}) and wrong params"
            )
    
    def _compute_param_score(
        self, 
        actual: dict[str, Any], 
        expected: dict[str, Any]
    ) -> float:
        """Compute a score for how well parameters match."""
        if not expected:
            return 1.0 if not actual else 0.5
        
        total_keys = set(expected.keys()) | set(actual.keys())
        if not total_keys:
            return 1.0
        
        matches = 0
        for key in expected:
            if key in actual:
                # Check if values match (with some flexibility for strings)
                expected_val = expected[key]
                actual_val = actual[key]
                
                if expected_val == actual_val:
                    matches += 1
                elif isinstance(expected_val, str) and isinstance(actual_val, str):
                    # Fuzzy string matching
                    if expected_val.lower() in actual_val.lower() or actual_val.lower() in expected_val.lower():
                        matches += 0.5
        
        return matches / len(expected) if expected else 1.0


class CompositeReward(RewardFunction):
    """
    Combines multiple reward functions with configurable weights.
    
    This allows for complex reward signals that consider multiple aspects
    of tool call quality.
    """
    
    def __init__(self, rewards: list[tuple[RewardFunction, float]]):
        """
        Args:
            rewards: List of (reward_function, weight) tuples
        """
        self.rewards = rewards
        total_weight = sum(w for _, w in rewards)
        self.rewards = [(r, w / total_weight) for r, w in rewards]  # Normalize weights
    
    def compute(
        self,
        tool_call: ToolCall,
        expected_tool: str,
        expected_args: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None
    ) -> VerifiableReward:
        total_score = 0.0
        reasons = []
        tool_correct = False
        params_correct = False
        
        for reward_fn, weight in self.rewards:
            result = reward_fn.compute(tool_call, expected_tool, expected_args, context)
            total_score += result.score * weight
            reasons.append(result.reason)
            if result.tool_correct:
                tool_correct = True
            if result.params_correct:
                params_correct = True
        
        return VerifiableReward(
            score=total_score,
            reason=" | ".join(reasons),
            tool_correct=tool_correct,
            params_correct=params_correct
        )


# === Helper functions for common reward computations ===

def verify_tool_call(
    tool_call: ToolCall,
    expected_tool: str,
    expected_args: dict[str, Any] | None = None,
    strict: bool = False
) -> VerifiableReward:
    """
    Convenience function to verify a tool call.
    
    Args:
        tool_call: The tool call to verify
        expected_tool: Expected tool name
        expected_args: Expected arguments
        strict: If True, requires exact parameter match
    
    Returns:
        VerifiableReward indicating correctness
    """
    if strict:
        reward_fn = ParameterAccuracyReward()
    else:
        reward_fn = ToolSelectionReward()
    
    return reward_fn.compute(tool_call, expected_tool, expected_args)
