"""
Trajectory Module

Defines the data structures for training trajectories with verifiable rewards.
A trajectory is a sequence of (input, tool_call, reward) that the model learns from.
"""

from typing import Any
from pydantic import BaseModel, Field
import json


class ToolCall(BaseModel):
    """
    Represents a tool call made by the model.
    
    This captures what tool was called and with what arguments.
    """
    name: str = Field(..., description="Name of the tool that was called")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    
    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tool call format."""
        return {
            "id": f"call_{hash(self.name + json.dumps(self.arguments, sort_keys=True)) % 10000:04d}",
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments)
            }
        }


class VerifiableReward(BaseModel):
    """
    A verifiable reward signal for RLVR training.
    
    This is one of the KEY INPUTS to the training process. Each trajectory
    is associated with a reward that can be objectively computed/verified.
    
    The reward indicates how good the tool call was:
    - 1.0 = Perfect tool selection and parameters
    - 0.5 = Partially correct (right tool, wrong params or vice versa)
    - 0.0 = Completely wrong
    
    For preference learning (DPO-style), we use chosen/rejected pairs.
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Reward score from 0 to 1")
    reason: str = Field(default="", description="Explanation of the reward")
    
    # Components that contribute to the reward
    tool_correct: bool = Field(default=False, description="Was the right tool selected?")
    params_correct: bool = Field(default=False, description="Were the parameters correct?")
    
    @classmethod
    def perfect(cls, reason: str = "Perfect tool call") -> "VerifiableReward":
        """Create a perfect reward (1.0)."""
        return cls(score=1.0, reason=reason, tool_correct=True, params_correct=True)
    
    @classmethod
    def partial(cls, reason: str, tool_ok: bool = True, params_ok: bool = False) -> "VerifiableReward":
        """Create a partial reward (0.5)."""
        return cls(score=0.5, reason=reason, tool_correct=tool_ok, params_correct=params_ok)
    
    @classmethod
    def wrong(cls, reason: str = "Wrong tool call") -> "VerifiableReward":
        """Create a zero reward."""
        return cls(score=0.0, reason=reason, tool_correct=False, params_correct=False)


class Trajectory(BaseModel):
    """
    A single training trajectory for RLVR.
    
    Contains:
    - The input prompt/context (what the user asked)
    - The tool call response (what the model did)
    - The verifiable reward (how good was it)
    
    For preference learning, we can also have a rejected_tool_call
    that represents a worse alternative.
    """
    # Input context
    system_prompt: str = Field(default="You are a helpful assistant that uses tools to accomplish tasks.")
    user_message: str = Field(..., description="The user's request")
    
    # Model response - the tool call
    tool_call: ToolCall = Field(..., description="The tool call made (chosen response)")
    rejected_tool_call: ToolCall | None = Field(default=None, description="A rejected/worse tool call for preference learning")
    
    # Verifiable reward signal
    reward: VerifiableReward = Field(..., description="The verifiable reward for this trajectory")
    
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    def to_openpipe_entry(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Convert to OpenPipe dataset entry format.
        
        This creates the format expected by OpenPipe's create_dataset_entries API.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_message},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [self.tool_call.to_openai_format()]
            }
        ]
        
        entry = {
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "metadata": {
                "reward_score": str(self.reward.score),
                "reward_reason": self.reward.reason,
                **{k: str(v) for k, v in self.metadata.items()}
            }
        }
        
        # Add rejected response for preference learning
        if self.rejected_tool_call:
            entry["rejected_message"] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [self.rejected_tool_call.to_openai_format()]
            }
        
        return entry


class TrajectoryGenerator:
    """
    Generates training trajectories from task definitions.
    
    This is a utility class to help create training data for RLVR.
    You can subclass this to create custom trajectory generators
    for your specific use case.
    """
    
    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or "You are a helpful assistant that uses tools to accomplish tasks."
    
    def create_trajectory(
        self,
        user_message: str,
        chosen_tool: str,
        chosen_args: dict[str, Any],
        reward: VerifiableReward,
        rejected_tool: str | None = None,
        rejected_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> Trajectory:
        """Create a single training trajectory."""
        trajectory = Trajectory(
            system_prompt=self.system_prompt,
            user_message=user_message,
            tool_call=ToolCall(name=chosen_tool, arguments=chosen_args),
            reward=reward,
            metadata=metadata or {}
        )
        
        # Add rejected alternative if provided (for preference learning)
        if rejected_tool is not None:
            trajectory.rejected_tool_call = ToolCall(
                name=rejected_tool, 
                arguments=rejected_args or {}
            )
        
        return trajectory
    
    def create_preference_pair(
        self,
        user_message: str,
        good_tool: str,
        good_args: dict[str, Any],
        bad_tool: str,
        bad_args: dict[str, Any],
        good_reason: str = "Correct tool selection",
        bad_reason: str = "Incorrect tool selection"
    ) -> Trajectory:
        """
        Create a preference pair for DPO-style training.
        
        This creates a trajectory with both a chosen (good) and rejected (bad) response,
        which is useful for preference optimization training.
        """
        return self.create_trajectory(
            user_message=user_message,
            chosen_tool=good_tool,
            chosen_args=good_args,
            reward=VerifiableReward.perfect(good_reason),
            rejected_tool=bad_tool,
            rejected_args=bad_args,
            metadata={"rejected_reason": bad_reason}
        )
