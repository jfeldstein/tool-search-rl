"""
Tool Definition Module

Provides a clean interface for defining available tools that the model
should learn to select from. Tools are defined in JSON Schema format
compatible with OpenAI/OpenPipe function calling.
"""

from typing import Any
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """A single parameter for a tool."""
    name: str
    type: str
    description: str
    required: bool = False
    enum: list[str] | None = None


class ToolDefinition(BaseModel):
    """
    Defines a single tool available for the model to call.
    
    Example:
        search_tool = ToolDefinition(
            name="web_search",
            description="Search the web for information",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query", required=True)
            ]
        )
    """
    name: str = Field(..., description="Unique identifier for the tool")
    description: str = Field(..., description="What the tool does - used by the model to decide when to use it")
    parameters: list[ToolParameter] = Field(default_factory=list)
    
    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI/OpenPipe function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }


class ToolRegistry(BaseModel):
    """
    Registry of all available tools for the model.
    
    This is one of the key inputs to the training process - it defines
    what tools the model should learn to select from.
    
    Example:
        registry = ToolRegistry(tools=[
            ToolDefinition(name="search", description="Search the web", ...),
            ToolDefinition(name="calculator", description="Do math", ...),
        ])
    """
    tools: list[ToolDefinition] = Field(default_factory=list)
    
    def add_tool(self, tool: ToolDefinition) -> None:
        """Add a tool to the registry."""
        self.tools.append(tool)
    
    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI/OpenPipe format."""
        return [tool.to_openai_schema() for tool in self.tools]
    
    @classmethod
    def from_dict_list(cls, tools: list[dict[str, Any]]) -> "ToolRegistry":
        """Create registry from a list of tool dictionaries."""
        return cls(tools=[ToolDefinition(**t) for t in tools])


# Example: Pre-built registry with common coding assistant tools
def create_coding_assistant_tools() -> ToolRegistry:
    """Create a registry of common coding assistant tools."""
    return ToolRegistry(tools=[
        ToolDefinition(
            name="read_file",
            description="Read the contents of a file at the specified path",
            parameters=[
                ToolParameter(name="path", type="string", description="The file path to read", required=True),
            ]
        ),
        ToolDefinition(
            name="write_file",
            description="Write content to a file, creating it if it doesn't exist",
            parameters=[
                ToolParameter(name="path", type="string", description="The file path to write to", required=True),
                ToolParameter(name="content", type="string", description="The content to write", required=True),
            ]
        ),
        ToolDefinition(
            name="search_code",
            description="Search for patterns in code files using regex",
            parameters=[
                ToolParameter(name="pattern", type="string", description="The regex pattern to search for", required=True),
                ToolParameter(name="directory", type="string", description="Directory to search in", required=False),
            ]
        ),
        ToolDefinition(
            name="run_command",
            description="Execute a shell command",
            parameters=[
                ToolParameter(name="command", type="string", description="The command to run", required=True),
                ToolParameter(name="working_dir", type="string", description="Working directory", required=False),
            ]
        ),
        ToolDefinition(
            name="list_files",
            description="List files and directories at the specified path",
            parameters=[
                ToolParameter(name="path", type="string", description="The directory path to list", required=True),
            ]
        ),
    ])
