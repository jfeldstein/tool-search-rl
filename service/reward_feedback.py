"""
Reward Feedback Module

Shows how to communicate verifiable rewards back to OpenPipe for training.

The flow:
1. Service receives user request
2. Model selects a tool (inference via OpenPipe)
3. Tool is executed, outcome is observed
4. Reward is computed based on outcome
5. Reward is reported back to OpenPipe -> Used for future training

This creates a continuous learning loop:
    User Request -> Model Inference -> Execution -> Reward -> Training Data
"""

import os
import json
import time
from typing import Any
from dataclasses import dataclass, field
from enum import Enum

from openpipe import OpenPipe


class RewardOutcome(Enum):
    """Possible outcomes for tool execution."""
    SUCCESS = "success"           # Tool executed correctly
    WRONG_TOOL = "wrong_tool"     # Wrong tool was selected
    WRONG_PARAMS = "wrong_params" # Right tool, wrong parameters
    PARTIAL = "partial"           # Partially correct
    ERROR = "error"               # Execution error


@dataclass
class ExecutionResult:
    """Result of executing a tool call."""
    tool_name: str
    arguments: dict[str, Any]
    outcome: RewardOutcome
    expected_tool: str | None = None
    expected_args: dict[str, Any] | None = None
    error_message: str | None = None
    execution_time_ms: float = 0


@dataclass 
class RewardSignal:
    """
    A verifiable reward signal to send back to OpenPipe.
    
    This is the KEY mechanism for RLVR - we observe the outcome
    and compute a reward that can be verified.
    """
    score: float                    # 0.0 to 1.0
    outcome: RewardOutcome
    reason: str
    tool_correct: bool = False
    params_correct: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_execution(cls, result: ExecutionResult) -> "RewardSignal":
        """Compute reward from execution result."""
        if result.outcome == RewardOutcome.SUCCESS:
            return cls(
                score=1.0,
                outcome=result.outcome,
                reason="Tool executed successfully with correct results",
                tool_correct=True,
                params_correct=True
            )
        elif result.outcome == RewardOutcome.WRONG_TOOL:
            return cls(
                score=0.0,
                outcome=result.outcome,
                reason=f"Selected {result.tool_name}, expected {result.expected_tool}",
                tool_correct=False,
                params_correct=False
            )
        elif result.outcome == RewardOutcome.WRONG_PARAMS:
            return cls(
                score=0.5,
                outcome=result.outcome,
                reason=f"Correct tool but wrong parameters",
                tool_correct=True,
                params_correct=False
            )
        elif result.outcome == RewardOutcome.PARTIAL:
            return cls(
                score=0.7,
                outcome=result.outcome,
                reason="Partially correct execution",
                tool_correct=True,
                params_correct=False
            )
        else:  # ERROR
            return cls(
                score=0.0,
                outcome=result.outcome,
                reason=f"Execution error: {result.error_message}",
                tool_correct=False,
                params_correct=False
            )


class RewardReporter:
    """
    Reports rewards back to OpenPipe for training data collection.
    
    OpenPipe uses these reward signals to:
    1. Build training datasets with outcome labels
    2. Enable preference learning (good vs bad responses)
    3. Continuous model improvement via RLVR
    """
    
    def __init__(self, api_key: str | None = None):
        self.client = OpenPipe(api_key=api_key or os.getenv("OPENPIPE_API_KEY"))
        self._pending_reports: list[dict] = []
    
    def report_reward(
        self,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        reward: RewardSignal,
        request_id: str | None = None
    ) -> dict[str, Any]:
        """
        Report a reward signal back to OpenPipe.
        
        This uses OpenPipe's report() API to log the interaction
        along with the computed reward for training.
        
        Args:
            request_payload: The original request sent to the model
            response_payload: The model's response
            reward: The computed reward signal
            request_id: Optional request ID for tracking
        
        Returns:
            Report response from OpenPipe
        """
        # Build tags that include reward information
        # These tags are used by OpenPipe to filter/organize training data
        tags = {
            "reward_score": reward.score,
            "reward_outcome": reward.outcome.value,
            "tool_correct": reward.tool_correct,
            "params_correct": reward.params_correct,
            # Custom tags for filtering
            "is_positive_example": reward.score >= 0.8,
            "is_negative_example": reward.score <= 0.2,
            "needs_review": 0.2 < reward.score < 0.8,
        }
        
        # Report to OpenPipe
        response = self.client.report(
            requested_at=time.time() * 1000,  # milliseconds
            received_at=time.time() * 1000,
            req_payload=request_payload,
            resp_payload=response_payload,
            status_code=200,
            tags=tags
        )
        
        return {
            "reported": True,
            "reward_score": reward.score,
            "outcome": reward.outcome.value,
            "reason": reward.reason,
            "response": response
        }
    
    def report_preference_pair(
        self,
        request_payload: dict[str, Any],
        chosen_response: dict[str, Any],
        rejected_response: dict[str, Any],
        chosen_reward: RewardSignal,
        rejected_reward: RewardSignal
    ) -> dict[str, Any]:
        """
        Report a preference pair for DPO-style training.
        
        This explicitly marks one response as better than another,
        which is ideal for preference optimization training.
        """
        # Report the chosen (good) response
        chosen_report = self.report_reward(
            request_payload=request_payload,
            response_payload=chosen_response,
            reward=chosen_reward
        )
        
        # Report the rejected (bad) response with low score
        rejected_report = self.report_reward(
            request_payload=request_payload,
            response_payload=rejected_response,
            reward=rejected_reward
        )
        
        return {
            "chosen": chosen_report,
            "rejected": rejected_report,
            "preference_logged": True
        }
    
    def batch_report(self, reports: list[tuple[dict, dict, RewardSignal]]) -> list[dict]:
        """Report multiple rewards in batch."""
        results = []
        for req, resp, reward in reports:
            result = self.report_reward(req, resp, reward)
            results.append(result)
        return results


# ============================================================
# INTEGRATION WITH SERVICE
# ============================================================

def compute_reward_from_outcome(
    tool_call: dict[str, Any],
    execution_result: Any,
    expected_outcome: Any = None
) -> RewardSignal:
    """
    Compute a verifiable reward based on execution outcome.
    
    This is where you define what makes a "good" tool selection.
    The reward should be objectively computable from the outcome.
    """
    tool_name = tool_call.get("name", "")
    arguments = tool_call.get("arguments", {})
    
    # Example: Check if execution succeeded
    if isinstance(execution_result, dict):
        if execution_result.get("success"):
            return RewardSignal(
                score=1.0,
                outcome=RewardOutcome.SUCCESS,
                reason="Execution successful",
                tool_correct=True,
                params_correct=True
            )
        elif execution_result.get("error"):
            return RewardSignal(
                score=0.0,
                outcome=RewardOutcome.ERROR,
                reason=f"Execution failed: {execution_result.get('error')}",
                tool_correct=False,
                params_correct=False
            )
    
    # Example: Check against expected outcome
    if expected_outcome:
        if tool_name == expected_outcome.get("tool"):
            if arguments == expected_outcome.get("arguments"):
                return RewardSignal(
                    score=1.0,
                    outcome=RewardOutcome.SUCCESS,
                    reason="Perfect match with expected",
                    tool_correct=True,
                    params_correct=True
                )
            else:
                return RewardSignal(
                    score=0.5,
                    outcome=RewardOutcome.WRONG_PARAMS,
                    reason="Correct tool, different parameters",
                    tool_correct=True,
                    params_correct=False
                )
        else:
            return RewardSignal(
                score=0.0,
                outcome=RewardOutcome.WRONG_TOOL,
                reason=f"Expected {expected_outcome.get('tool')}, got {tool_name}",
                tool_correct=False,
                params_correct=False
            )
    
    # Default: partial reward if we can't fully verify
    return RewardSignal(
        score=0.5,
        outcome=RewardOutcome.PARTIAL,
        reason="Outcome could not be fully verified",
        tool_correct=True,
        params_correct=False
    )


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_reward_reporting():
    """
    Example showing the complete reward reporting flow.
    """
    print("=" * 60)
    print("REWARD REPORTING EXAMPLE")
    print("=" * 60)
    
    # Simulated request/response (normally from actual inference)
    request_payload = {
        "model": "openpipe:tool-selector-llama-8b-v1",
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Show me the contents of main.py"}
        ],
        "tools": [
            {"type": "function", "function": {"name": "read_file", "parameters": {}}},
            {"type": "function", "function": {"name": "write_file", "parameters": {}}},
        ]
    }
    
    response_payload = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "main.py"}'
                    }
                }]
            }
        }]
    }
    
    # Compute reward based on outcome
    tool_call = {
        "name": "read_file",
        "arguments": {"path": "main.py"}
    }
    
    # Simulate execution result
    execution_result = {"success": True, "content": "# main.py content..."}
    
    # Compute reward
    reward = compute_reward_from_outcome(
        tool_call=tool_call,
        execution_result=execution_result
    )
    
    print(f"\n📊 Computed Reward:")
    print(f"   Score: {reward.score}")
    print(f"   Outcome: {reward.outcome.value}")
    print(f"   Reason: {reward.reason}")
    print(f"   Tool Correct: {reward.tool_correct}")
    print(f"   Params Correct: {reward.params_correct}")
    
    # Report to OpenPipe (would actually call API in production)
    print(f"\n📤 Reporting to OpenPipe:")
    print(f"   Request: {request_payload['messages'][-1]['content']}")
    print(f"   Tool Selected: {tool_call['name']}")
    print(f"   Reward Score: {reward.score}")
    print(f"   Tags: reward_score={reward.score}, outcome={reward.outcome.value}")
    
    print("\n" + "=" * 60)
    print("This reward data flows back to OpenPipe for training!")
    print("=" * 60)


if __name__ == "__main__":
    example_reward_reporting()
