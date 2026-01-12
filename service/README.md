# Tool Selection Service - Hello World

A containerized service that uses an OpenPipe RLVR-trained model to select appropriate tools for user requests.

```
┌─────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   LOCAL (Container)              CLOUD (OpenPipe)               │
│   ┌─────────────────┐            ┌─────────────────┐            │
│   │                 │            │                 │            │
│   │  FastAPI        │  ──────►   │  Your Trained   │            │
│   │  Service        │  request   │  Model          │            │
│   │                 │            │  (RLVR)         │            │
│   │  - Tool defs    │  ◄──────   │                 │            │
│   │  - Execution    │  response  │  Inference API  │            │
│   │                 │            │                 │            │
│   └─────────────────┘            └─────────────────┘            │
│         ▲                                                        │
│         │                                                        │
│   ┌─────┴─────┐                                                  │
│   │   User    │                                                  │
│   │  Request  │                                                  │
│   └───────────┘                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Build and Run (Demo Mode - No API Key)

```bash
# Build the container
docker build -t tool-selector .

# Run in demo mode
docker run -p 8000:8000 tool-selector
```

### 2. Test the Service

```bash
# Health check
curl http://localhost:8000/

# Demo mode (no API key needed)
curl -X POST http://localhost:8000/demo/select-tool \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me the contents of main.py"}'
```

### 3. Production Mode (With Trained Model)

```bash
# Set your API key
export OPENPIPE_API_KEY=your-key-here
export OPENPIPE_MODEL_SLUG=tool-selector-llama-8b-v1

# Run with docker-compose
docker-compose up --build
```

## API Endpoints

### `GET /` - Service Info
```bash
curl http://localhost:8000/
```

Response:
```json
{
  "status": "healthy",
  "service": "tool-selection-service",
  "model": "tool-selector-llama-8b-v1",
  "tools": ["read_file", "list_files", "search_code"]
}
```

### `POST /select-tool` - Select Tool (Production)
Uses your RLVR-trained model via OpenPipe.

```bash
curl -X POST http://localhost:8000/select-tool \
  -H "Content-Type: application/json" \
  -d '{"message": "What files are in the src directory?"}'
```

Response:
```json
{
  "user_message": "What files are in the src directory?",
  "selected_tool": {
    "name": "list_files",
    "arguments": {"path": "src"}
  },
  "model_used": "tool-selector-llama-8b-v1"
}
```

### `POST /demo/select-tool` - Demo Mode
Works without API key using rule-based selection.

```bash
curl -X POST http://localhost:8000/demo/select-tool \
  -H "Content-Type: application/json" \
  -d '{"message": "Find all TODO comments"}'
```

### `POST /execute` - Select + Execute
Selects tool AND executes it (simulated).

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"message": "Read the config.json file"}'
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENPIPE_API_KEY` | Your OpenPipe API key | Yes (for production) |
| `OPENPIPE_MODEL_SLUG` | Your trained model slug | No (has default) |
| `ENVIRONMENT` | development/production | No |

## Integration with Training

This service uses models trained via the RLVR training pipeline:

```python
# In your training code
from rlvr_tool_training.config import TrainingConfig

config = TrainingConfig(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    openpipe={
        "model_slug": "tool-selector-llama-8b-v1"  # <-- Use this in service
    }
)
```

Then in the service, reference the same slug:
```bash
export OPENPIPE_MODEL_SLUG=tool-selector-llama-8b-v1
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Or with uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
