"""
Tool Selection Verification Module

This is the core of the POC: verifying whether the model selected
the RIGHT tool at the RIGHT time.

The key insight: We can objectively verify tool SELECTION correctness
without executing the tool. This makes the reward signal verifiable.

Verification approaches:
1. Ground truth comparison - compare against known-correct tool
2. Intent matching - does the tool match the user's intent?
3. Rule-based verification - heuristics for common patterns
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field


@dataclass
class ToolSelectionQuery:
    """
    A query for which the model should select a tool.
    
    This represents the INPUT to the tool selection decision.
    """
    user_message: str
    available_tools: list[str]
    context: dict[str, Any] | None = None
    
    # Optional: ground truth for verification
    correct_tool: str | None = None
    correct_args: dict[str, Any] | None = None


@dataclass
class ToolSelectionResponse:
    """
    The model's tool selection response.
    
    This represents the OUTPUT of the tool selection decision.
    """
    selected_tool: str
    selected_args: dict[str, Any]
    reasoning: str | None = None


@dataclass  
class SelectionVerdict:
    """
    Verdict on whether a tool selection was correct.
    
    This is the VERIFIABLE REWARD signal for RLVR.
    """
    correct: bool
    score: float  # 0.0 to 1.0
    reason: str
    
    # Breakdown
    tool_correct: bool = False
    args_correct: bool = False
    
    @classmethod
    def right_tool(cls, reason: str = "Correct tool selected") -> SelectionVerdict:
        return cls(correct=True, score=1.0, reason=reason, tool_correct=True, args_correct=True)
    
    @classmethod
    def wrong_tool(cls, reason: str = "Wrong tool selected") -> SelectionVerdict:
        return cls(correct=False, score=0.0, reason=reason, tool_correct=False, args_correct=False)
    
    @classmethod
    def right_tool_wrong_args(cls, reason: str = "Correct tool, incorrect arguments") -> SelectionVerdict:
        return cls(correct=False, score=0.5, reason=reason, tool_correct=True, args_correct=False)


class ToolSelectionVerifier(ABC):
    """
    Abstract base class for verifying tool selection correctness.
    
    Implement this to define how to verify if a tool selection was right.
    """
    
    @abstractmethod
    def verify(
        self,
        query: ToolSelectionQuery,
        response: ToolSelectionResponse,
    ) -> SelectionVerdict:
        """
        Verify if the tool selection was correct.
        
        Args:
            query: The original query (what the user asked)
            response: The model's selection (what tool was chosen)
        
        Returns:
            SelectionVerdict with score and explanation
        """
        pass


class GroundTruthVerifier(ToolSelectionVerifier):
    """
    Verifier that compares against known ground truth.
    
    Use this when you have labeled data with correct tool selections.
    """
    
    def __init__(self, check_args: bool = True, fuzzy_args: bool = True):
        """
        Args:
            check_args: Whether to verify arguments too
            fuzzy_args: Allow partial argument matching
        """
        self.check_args = check_args
        self.fuzzy_args = fuzzy_args
    
    def verify(
        self,
        query: ToolSelectionQuery,
        response: ToolSelectionResponse,
    ) -> SelectionVerdict:
        if query.correct_tool is None:
            return SelectionVerdict(
                correct=False,
                score=0.5,
                reason="No ground truth available",
            )
        
        # Check tool name
        tool_correct = response.selected_tool == query.correct_tool
        
        if not tool_correct:
            return SelectionVerdict.wrong_tool(
                f"Selected '{response.selected_tool}', expected '{query.correct_tool}'"
            )
        
        # Check arguments if required
        if not self.check_args or query.correct_args is None:
            return SelectionVerdict.right_tool("Correct tool selected")
        
        args_correct = self._check_args(response.selected_args, query.correct_args)
        
        if args_correct:
            return SelectionVerdict.right_tool("Correct tool and arguments")
        else:
            return SelectionVerdict.right_tool_wrong_args(
                f"Correct tool, but args differ: got {response.selected_args}, expected {query.correct_args}"
            )
    
    def _check_args(self, actual: dict, expected: dict) -> bool:
        """Check if arguments match."""
        if not self.fuzzy_args:
            return actual == expected
        
        # Fuzzy matching: check required keys exist with similar values
        for key, expected_val in expected.items():
            if key not in actual:
                return False
            actual_val = actual[key]
            
            # String comparison is fuzzy
            if isinstance(expected_val, str) and isinstance(actual_val, str):
                if expected_val.lower() not in actual_val.lower() and actual_val.lower() not in expected_val.lower():
                    return False
            elif actual_val != expected_val:
                return False
        
        return True


class IntentMatchingVerifier(ToolSelectionVerifier):
    """
    Verifier that checks if the tool matches the user's intent.
    
    Uses pattern matching to determine if the selected tool is
    appropriate for the type of request.
    """
    
    def __init__(self):
        # Define intent patterns and valid tools for each
        self.intent_patterns: list[tuple[str, list[str], list[str]]] = [
            # (pattern_name, regex_patterns, valid_tools)
            ("read_intent", [r"\bread\b", r"\bshow\b", r"\bview\b", r"\bcat\b", r"\bcontents?\b", r"\bwhat'?s in\b"], 
             ["read_file", "cat", "view"]),
            ("write_intent", [r"\bwrite\b", r"\bcreate\b", r"\bsave\b", r"\bmake\b.*file"], 
             ["write_file", "create_file", "save"]),
            ("search_intent", [r"\bsearch\b", r"\bfind\b", r"\bgrep\b", r"\blook for\b", r"\bwhere\b.*\?"], 
             ["search_code", "grep", "find", "search"]),
            ("list_intent", [r"\blist\b", r"\bls\b", r"\bfiles in\b", r"\bdirectory\b", r"\bfolder\b"], 
             ["list_files", "ls", "list_directory"]),
            ("run_intent", [r"\brun\b", r"\bexecute\b", r"\btest\b", r"\bpytest\b", r"\bnpm\b"], 
             ["run_command", "execute", "shell", "run_tests"]),
            ("edit_intent", [r"\bedit\b", r"\bmodify\b", r"\bchange\b", r"\bupdate\b", r"\bfix\b"], 
             ["edit_file", "modify", "patch", "update_file"]),
        ]
    
    def verify(
        self,
        query: ToolSelectionQuery,
        response: ToolSelectionResponse,
    ) -> SelectionVerdict:
        message_lower = query.user_message.lower()
        
        # Find matching intent
        matched_intent = None
        valid_tools = []
        
        for intent_name, patterns, tools in self.intent_patterns:
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    matched_intent = intent_name
                    valid_tools = tools
                    break
            if matched_intent:
                break
        
        if not matched_intent:
            # No clear intent detected - can't verify
            return SelectionVerdict(
                correct=True,  # Give benefit of doubt
                score=0.5,
                reason="No clear intent pattern detected",
            )
        
        # Check if selected tool matches intent
        tool_matches = any(
            response.selected_tool.lower() in valid.lower() or valid.lower() in response.selected_tool.lower()
            for valid in valid_tools
        )
        
        if tool_matches:
            return SelectionVerdict.right_tool(
                f"Tool '{response.selected_tool}' matches {matched_intent}"
            )
        else:
            return SelectionVerdict.wrong_tool(
                f"Tool '{response.selected_tool}' doesn't match {matched_intent}. Expected one of: {valid_tools}"
            )


class RuleBasedVerifier(ToolSelectionVerifier):
    """
    Verifier using custom rules.
    
    Add rules that encode domain knowledge about when tools should be used.
    """
    
    def __init__(self):
        self.rules: list[Callable[[ToolSelectionQuery, ToolSelectionResponse], SelectionVerdict | None]] = []
    
    def add_rule(
        self,
        rule: Callable[[ToolSelectionQuery, ToolSelectionResponse], SelectionVerdict | None]
    ) -> None:
        """Add a verification rule. Return None to skip, SelectionVerdict to decide."""
        self.rules.append(rule)
    
    def verify(
        self,
        query: ToolSelectionQuery,
        response: ToolSelectionResponse,
    ) -> SelectionVerdict:
        for rule in self.rules:
            verdict = rule(query, response)
            if verdict is not None:
                return verdict
        
        # No rule matched
        return SelectionVerdict(
            correct=True,
            score=0.5,
            reason="No rule matched - inconclusive",
        )


class CompositeSelectionVerifier(ToolSelectionVerifier):
    """
    Combines multiple verifiers with priority ordering.
    
    Uses the first verifier that gives a definitive answer.
    """
    
    def __init__(self, verifiers: list[ToolSelectionVerifier]):
        self.verifiers = verifiers
    
    def verify(
        self,
        query: ToolSelectionQuery,
        response: ToolSelectionResponse,
    ) -> SelectionVerdict:
        for verifier in self.verifiers:
            verdict = verifier.verify(query, response)
            # Use this verdict if it's definitive (not 0.5 score)
            if verdict.score != 0.5:
                return verdict
        
        # All inconclusive - return last verdict
        return self.verifiers[-1].verify(query, response) if self.verifiers else SelectionVerdict(
            correct=True, score=0.5, reason="No verifiers configured"
        )


# ============================================================
# TRAINING DATA GENERATION
# ============================================================

@dataclass
class ToolSelectionExample:
    """
    A single training example for tool selection.
    
    This is what gets sent to OpenPipe for training.
    """
    query: ToolSelectionQuery
    response: ToolSelectionResponse
    verdict: SelectionVerdict
    
    # For preference learning: a worse alternative
    rejected_response: ToolSelectionResponse | None = None
    
    def to_training_entry(self, tool_schemas: list[dict]) -> dict:
        """Convert to OpenPipe training entry format."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Select the most appropriate tool for the user's request."},
            {"role": "user", "content": self.query.user_message},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": self.response.selected_tool,
                        "arguments": str(self.response.selected_args)
                    }
                }]
            }
        ]
        
        entry = {
            "messages": messages,
            "tools": tool_schemas,
            "tool_choice": "auto",
            "metadata": {
                "reward_score": str(self.verdict.score),
                "tool_correct": str(self.verdict.tool_correct),
                "verification_reason": self.verdict.reason,
            }
        }
        
        # Add rejected response for preference learning
        if self.rejected_response:
            entry["rejected_message"] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_002",
                    "type": "function",
                    "function": {
                        "name": self.rejected_response.selected_tool,
                        "arguments": str(self.rejected_response.selected_args)
                    }
                }]
            }
        
        return entry


def create_selection_example(
    user_message: str,
    available_tools: list[str],
    selected_tool: str,
    selected_args: dict[str, Any],
    correct_tool: str,
    correct_args: dict[str, Any] | None = None,
    rejected_tool: str | None = None,
    rejected_args: dict[str, Any] | None = None,
) -> ToolSelectionExample:
    """
    Convenience function to create a training example.
    
    Args:
        user_message: What the user asked
        available_tools: Tools the model could choose from
        selected_tool: Tool the model chose
        selected_args: Arguments the model provided
        correct_tool: The actually correct tool
        correct_args: The correct arguments (optional)
        rejected_tool: A wrong tool choice (for preference learning)
        rejected_args: Arguments for the wrong choice
    """
    query = ToolSelectionQuery(
        user_message=user_message,
        available_tools=available_tools,
        correct_tool=correct_tool,
        correct_args=correct_args,
    )
    
    response = ToolSelectionResponse(
        selected_tool=selected_tool,
        selected_args=selected_args,
    )
    
    # Verify
    verifier = GroundTruthVerifier()
    verdict = verifier.verify(query, response)
    
    # Create example
    example = ToolSelectionExample(
        query=query,
        response=response,
        verdict=verdict,
    )
    
    if rejected_tool:
        example.rejected_response = ToolSelectionResponse(
            selected_tool=rejected_tool,
            selected_args=rejected_args or {},
        )
    
    return example
