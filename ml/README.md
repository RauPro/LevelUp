# ML Module - AI Agent for Programming Problems

This module contains an AI-powered agent that can generate programming problems and solve them iteratively.

## Features

- **Problem Generation**: Uses Mistral AI to generate programming problems based on user constraints
- **Code Generation**: Automatically writes Python code to solve the generated problems
- **Sandbox Testing**: Tests the code in a secure E2B sandbox environment
- **Iterative Improvement**: Attempts to fix code up to 5 times, or regenerate the problem if needed
- **LangGraph Workflow**: Uses LangGraph for orchestrating the multi-step process

## Setup

### 1. Install Dependencies

The required dependencies should already be installed in your virtual environment. If not, run:

```bash
uv pip install langgraph langchain-mistralai langchain-core e2b-code-interpreter
```

### 2. Get API Keys

You'll need API keys for:

- **Mistral AI**: Sign up at [https://mistral.ai/](https://mistral.ai/)
- **E2B**: Sign up at [https://e2b.dev/](https://e2b.dev/)

### 3. Set Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
E2B_API_KEY=your_e2b_api_key_here
```

Or export them in your shell:

```bash
export MISTRAL_API_KEY="your_mistral_api_key_here"
export E2B_API_KEY="your_e2b_api_key_here"
```

## Usage

### Running the Agent

```bash
python ml/agent.py
```

### Using in Your Code

```python
from ml.agent import app, SessionState

# Create a session with your problem constraints
state = SessionState("Generate a problem involving arrays and sorting")

# Run the workflow
final_state = app.invoke(state)

# Check results
print(f"Tests passed: {final_state.tests_passed}")
print(f"Final code: {final_state.code}")
```

## Workflow Steps

1. **Problem Agent**: Generates a programming problem with test cases
2. **Code Agent**: Writes Python code to solve the problem
3. **Test Runner**: Executes the code in a sandbox with test inputs
4. **Decision Logic**:
   - If tests pass → End successfully
   - If tests fail and < 5 code attempts → Try coding again
   - If tests fail and ≥ 5 code attempts and < 2 problem attempts → Generate new problem
   - If ≥ 2 problem attempts → End with failure

## Testing

Run the test to verify everything is set up correctly:

```bash
python test_agent.py
```

This will test the workflow compilation without requiring API keys.

## File Structure

- `agent.py` - Main agent implementation with LangGraph workflow
- `../test_agent.py` - Test script for verifying setup
- `../.env.example` - Template for environment variables

## Troubleshooting

### Import Errors
- Make sure all dependencies are installed: `uv pip install langgraph langchain-mistralai langchain-core e2b-code-interpreter`

### API Key Errors
- Verify your API keys are correct and have sufficient credits
- Check that environment variables are properly set

### Sandbox Errors
- E2B requires a valid API key and sufficient credits
- Make sure your network allows connections to E2B servers
