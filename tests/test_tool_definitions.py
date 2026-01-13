"""Tests for tool definitions module."""

import pytest
from rlvr_tool_training.config import ToolDefinition, ToolParameter, ToolRegistry
from rlvr_tool_training.config.tool_definitions import create_coding_assistant_tools


class TestToolParameter:
    def test_basic_parameter(self):
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True
        )
        assert param.name == "query"
        assert param.type == "string"
        assert param.required is True
    
    def test_parameter_with_enum(self):
        param = ToolParameter(
            name="format",
            type="string",
            description="Output format",
            enum=["json", "csv", "text"]
        )
        assert param.enum == ["json", "csv", "text"]


class TestToolDefinition:
    def test_basic_tool(self):
        tool = ToolDefinition(
            name="search",
            description="Search for things",
            parameters=[]
        )
        assert tool.name == "search"
        assert tool.description == "Search for things"
    
    def test_to_openai_schema(self):
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="File path",
                    required=True
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="File encoding",
                    required=False
                )
            ]
        )
        
        schema = tool.to_openai_schema()
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_file"
        assert schema["function"]["description"] == "Read a file"
        assert "path" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["path"]


class TestToolRegistry:
    def test_add_and_get_tool(self):
        registry = ToolRegistry()
        tool = ToolDefinition(name="test_tool", description="Test")
        
        registry.add_tool(tool)
        
        retrieved = registry.get_tool("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"
    
    def test_get_nonexistent_tool(self):
        registry = ToolRegistry()
        assert registry.get_tool("nonexistent") is None
    
    def test_to_openai_tools(self):
        registry = ToolRegistry(tools=[
            ToolDefinition(name="tool1", description="Tool 1"),
            ToolDefinition(name="tool2", description="Tool 2"),
        ])
        
        openai_tools = registry.to_openai_tools()
        
        assert len(openai_tools) == 2
        assert all(t["type"] == "function" for t in openai_tools)
    
    def test_from_dict_list(self):
        tools_data = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"},
        ]
        
        registry = ToolRegistry.from_dict_list(tools_data)
        
        assert len(registry.tools) == 2
        assert registry.get_tool("tool1") is not None


class TestCodingAssistantTools:
    def test_creates_expected_tools(self):
        tools = create_coding_assistant_tools()
        
        assert len(tools.tools) >= 5
        
        tool_names = [t.name for t in tools.tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "search_code" in tool_names
        assert "run_command" in tool_names
        assert "list_files" in tool_names
    
    def test_tools_have_valid_schemas(self):
        tools = create_coding_assistant_tools()
        
        for tool in tools.tools:
            schema = tool.to_openai_schema()
            assert "type" in schema
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
