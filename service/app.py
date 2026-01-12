"""
Hello World Service - Uses OpenPipe-trained model for tool selection.

This service runs locally (containerized) and calls the OpenPipe cloud
for inference using your fine-tuned model.

Training: OpenPipe Cloud (via rlvr_tool_training)
Inference: This local service -> OpenPipe API -> Your trained model
"""

import os
import json
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

# Your OpenPipe-trained model slug (from training)
MODEL_SLUG = os.getenv("OPENPIPE_MODEL_SLUG", "tool-selector-llama-8b-v1")

# Available tools this service can execute
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the specified path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at the specified path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The directory path to list"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for patterns in code files using regex",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The regex pattern to search for"},
                    "directory": {"type": "string", "description": "Directory to search in"}
                },
                "required": ["pattern"]
            }
        }
    },
]


# ============================================================
# OPENPIPE CLIENT SETUP
# ============================================================

# Global client instance
openpipe_client = None


def get_openpipe_client():
    """Get or create the OpenPipe client."""
    global openpipe_client
    if openpipe_client is None:
        from openpipe import OpenPipe
        openpipe_client = OpenPipe(
            api_key=os.getenv("OPENPIPE_API_KEY")
        )
    return openpipe_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize client on startup."""
    print(f"🚀 Starting service with model: {MODEL_SLUG}")
    print(f"📦 Available tools: {[t['function']['name'] for t in AVAILABLE_TOOLS]}")
    yield
    print("👋 Shutting down service")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Tool Selection Service",
    description="A service that uses an RLVR-trained model for intelligent tool selection",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class UserRequest(BaseModel):
    """User request to the service."""
    message: str = Field(..., description="The user's request/question")
    context: dict[str, Any] = Field(default_factory=dict, description="Optional context")


class ToolCall(BaseModel):
    """A tool call selected by the model."""
    name: str
    arguments: dict[str, Any]


class ServiceResponse(BaseModel):
    """Response from the service."""
    user_message: str
    selected_tool: ToolCall | None
    model_used: str
    raw_response: dict[str, Any] | None = None


# ============================================================
# TOOL EXECUTION (SIMULATED)
# ============================================================

def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a tool call (simulated for this hello world example).
    
    In a real service, this would actually perform the operations.
    """
    if tool_name == "read_file":
        return f"[Simulated] Contents of {arguments.get('path', 'unknown')}: Hello, World!"
    elif tool_name == "list_files":
        return f"[Simulated] Files in {arguments.get('path', '.')}: file1.py, file2.py, README.md"
    elif tool_name == "search_code":
        return f"[Simulated] Found 3 matches for '{arguments.get('pattern', '')}'"
    else:
        return f"[Simulated] Executed {tool_name} with {arguments}"


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "tool-selection-service",
        "model": MODEL_SLUG,
        "tools": [t["function"]["name"] for t in AVAILABLE_TOOLS]
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/select-tool", response_model=ServiceResponse)
async def select_tool(request: UserRequest):
    """
    Use the RLVR-trained model to select the appropriate tool for the user's request.
    
    This endpoint:
    1. Sends the user message to the OpenPipe-trained model
    2. Model selects the best tool and parameters
    3. Returns the tool selection (and optionally executes it)
    """
    try:
        # Get the OpenPipe client (wraps OpenAI-compatible API)
        client = get_openpipe_client()
        
        # Call the trained model via OpenPipe
        # The model slug references your RLVR-trained model
        response = client.chat.completions.create(
            model=f"openpipe:{MODEL_SLUG}",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant. Use the provided tools to help users with their requests. Always select the most appropriate tool."
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            tools=AVAILABLE_TOOLS,
            tool_choice="auto",
            # OpenPipe-specific: tag this request for monitoring
            openpipe={
                "tags": {
                    "service": "tool-selection",
                    "environment": os.getenv("ENVIRONMENT", "development")
                }
            }
        )
        
        # Extract tool call from response
        message = response.choices[0].message
        selected_tool = None
        
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            selected_tool = ToolCall(
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments)
            )
        
        return ServiceResponse(
            user_message=request.message,
            selected_tool=selected_tool,
            model_used=MODEL_SLUG,
            raw_response={
                "finish_reason": response.choices[0].finish_reason,
                "content": message.content
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute")
async def execute_request(request: UserRequest):
    """
    Full pipeline: Select tool AND execute it.
    
    1. Model selects the tool
    2. Service executes the tool
    3. Returns the result
    """
    # First, get the tool selection
    selection = await select_tool(request)
    
    result = None
    if selection.selected_tool:
        # Execute the selected tool
        result = execute_tool(
            selection.selected_tool.name,
            selection.selected_tool.arguments
        )
    
    return {
        "user_message": request.message,
        "selected_tool": selection.selected_tool,
        "execution_result": result,
        "model_used": selection.model_used
    }


# ============================================================
# DEMO MODE (No API key required)
# ============================================================

@app.post("/demo/select-tool", response_model=ServiceResponse)
async def demo_select_tool(request: UserRequest):
    """
    Demo endpoint that simulates model responses without calling OpenPipe.
    
    Useful for testing the service without an API key.
    """
    # Simple rule-based selection for demo
    message_lower = request.message.lower()
    
    if any(word in message_lower for word in ["read", "show", "contents", "what's in"]):
        # Extract a filename if present
        path = "main.py"  # default
        if ".py" in message_lower or ".txt" in message_lower or ".json" in message_lower:
            words = request.message.split()
            for word in words:
                if "." in word:
                    path = word.strip("'\"")
                    break
        
        selected_tool = ToolCall(name="read_file", arguments={"path": path})
    
    elif any(word in message_lower for word in ["list", "files", "directory", "folder"]):
        path = "."
        if "/" in request.message or "src" in message_lower:
            path = "src"
        selected_tool = ToolCall(name="list_files", arguments={"path": path})
    
    elif any(word in message_lower for word in ["search", "find", "grep", "pattern"]):
        pattern = "TODO"  # default
        selected_tool = ToolCall(name="search_code", arguments={"pattern": pattern, "directory": "."})
    
    else:
        selected_tool = None
    
    return ServiceResponse(
        user_message=request.message,
        selected_tool=selected_tool,
        model_used="demo-mode",
        raw_response={"note": "This is demo mode - no actual model called"}
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
