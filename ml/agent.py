"""
AI-Powered Programming Problem Generator and Solver

This module implements a LangGraph workflow that:
1. Generates programming problems based on user user_prompt
2. Writes code to solve the problems
3. Tests the code in a sandbox environment
4. Iteratively improves until tests pass or max attempts reached

Dependencies:
- langgraph: For workflow orchestration
- langchain-mistralai: For LLM integration
- langchain-core: For core LangChain functionality
- e2b-code-interpreter: For code execution sandbox
- pydantic: For data validation

Environment Variables Required:
- MISTRAL_API_KEY: Your Mistral AI API key
- E2B_API_KEY: Your E2B sandbox API key
"""

import os
import re
from typing import Optional, TypedDict

from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox  # type: ignore
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.api.schemas import DifficultyLevel, ProblemTopic

# Load environment variables from .env file
load_dotenv()


# Define the Problem structure
class Problem(BaseModel):
    description: str
    tests: list[dict[str, str]]


# Define the state to track progress using TypedDict for LangGraph compatibility
class SessionState(TypedDict):
    user_prompt: str
    topic: ProblemTopic
    difficulty: DifficultyLevel
    problem: Optional[Problem]
    code: Optional[str]
    tests_passed: bool
    code_attempts: int
    problem_attempts: int


def create_initial_state(user_prompt: str, topic: ProblemTopic, difficulty: DifficultyLevel) -> SessionState:
    """Create initial state dictionary."""
    return SessionState(user_prompt=user_prompt, topic=topic, difficulty=difficulty, problem=None, code=None, tests_passed=False, code_attempts=0, problem_attempts=0)


# Initialize LLM and sandbox (in practice, add your API keys)
# Make sure to set your MISTRAL_API_KEY environment variable
llm = ChatMistralAI(model_name="mistral-tiny")
# Note: E2B requires an API key, set E2B_API_KEY environment variable
sandbox = Sandbox(api_key=os.getenv("E2B_API_KEY"))


# Node functions
def problem_agent(state: SessionState) -> SessionState:
    prompt = ChatPromptTemplate.from_template(
        "Generate a programming problem with User input: {user_prompt}. "
        "Topic from user  prompt: {topic}. "
        "Difficulty level from user prompt: {difficulty}. "
        "The problem should be solvable with a Python function that reads from stdin and writes to stdout. "
        "For inputs with multiple values, use space-separated format on a single line. "
        "Return JSON with 'description' (string) and 'tests' (array of objects with 'input' and 'output' as strings). "
        "Example: {{'description': 'Add two space-separated numbers', 'tests': [{{'input': '5 3', 'output': '8'}}]}}"
    )

    # Instead of using PydanticOutputParser, let's handle JSON manually
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"user_prompt": state["user_prompt"], "topic": state["topic"].value, "difficulty": state["difficulty"].value})
        # Try to extract JSON from the response
        import json
        import re

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            problem_data = json.loads(json_str)
            problem = Problem(description=problem_data["description"], tests=problem_data["tests"])
        else:
            # Fallback: create a simple math problem
            problem = Problem(description="Add two space-separated integers", tests=[{"input": "5 3", "output": "8"}])

    except Exception as e:
        print(f"Error parsing problem: {e}")
        # Fallback: create a simple math problem
        problem = Problem(description="Add two space-separated integers", tests=[{"input": "5 3", "output": "8"}])

    state["problem"] = problem
    state["code_attempts"] = 0  # Reset for new problem
    state["problem_attempts"] += 1
    return state


def code_agent(state: SessionState) -> SessionState:
    if state["problem"] is None:
        raise ValueError("Problem must be set before generating code")

    prompt = ChatPromptTemplate.from_template("Write a Python solution for: {problem_description}. The solution should read input from stdin and write output to stdout. If the input contains multiple values, they will be space-separated on a single line. Use input().split() to parse space-separated values. Return ONLY the Python code, no markdown formatting, no explanations. Example for adding two numbers: a, b = map(int, input().split())\nprint(a + b)")
    chain = prompt | llm | StrOutputParser()
    code = chain.invoke({"problem_description": state["problem"].description})

    # Clean the code by removing markdown formatting and extra text
    if "```python" in code:
        # Extract code from markdown
        code_match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1)
    elif "```" in code:
        # Remove any other code blocks
        code = re.sub(r"```.*?\n(.*?)\n```", r"\1", code, flags=re.DOTALL)

    # Remove common explanatory text
    lines = code.split("\n")
    clean_lines = []
    for line in lines:
        if not line.strip().startswith("#") or "import" in line or "def" in line:
            clean_lines.append(line)

    code = "\n".join(clean_lines).strip()

    state["code"] = code
    state["code_attempts"] += 1
    return state


def run_tests(state: SessionState) -> SessionState:
    """Run tests using E2B sandbox if available, otherwise use local execution."""

    if state["code"] is None:
        state["tests_passed"] = False
        return state

    if state["problem"] is None:
        state["tests_passed"] = False
        return state

    code = state["code"]
    test = state["problem"].tests[0]  # Single test for simplicity

    # Clean the code by removing markdown formatting
    if "```python" in code:
        code_match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1)
    elif "```" in code:
        code = re.sub(r"```.*?\n(.*?)\n```", r"\1", code, flags=re.DOTALL)
    print(f"Running code:\n{code}\nWith test input: '{test['input']}' and expected output: '{test['output']}'")
    # Try E2B sandbox first, fallback to local execution
    try:
        # Try E2B sandbox
        try:
            test_input = test["input"]
            sandbox.files.write("tests.txt", test_input)
            execution = sandbox.run_code(code, input_files=["tests.txt"])
            print(f"E2B execution logs: {execution.logs}")

            if execution.logs.stdout:
                output = execution.logs.stdout[0].strip()
                expected = test["output"].strip()
                state["tests_passed"] = output == expected
                print(f"Expected: '{expected}', Got: '{output}', Passed: {state['tests_passed']}")
            else:
                state["tests_passed"] = False
                print("No stdout output from E2B execution.")

        except Exception as e2b_error:
            print(f"E2B execution failed: {e2b_error}, falling back to local execution...")

            # Fallback to local execution
            import os
            import subprocess
            import sys
            import tempfile

            print(f"Running test locally with input: '{test['input']}'")

            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Run the code with test input
            process = subprocess.run([sys.executable, temp_file], input=test["input"], capture_output=True, text=True, timeout=5)

            # Clean up temp file
            os.unlink(temp_file)

            # Check results
            if process.returncode == 0:
                output = process.stdout.strip()
                expected = test["output"].strip()
                state["tests_passed"] = output == expected
                print(f"Expected: '{expected}', Got: '{output}', Passed: {state['tests_passed']}")
            else:
                state["tests_passed"] = False
                print(f"Local execution failed: {process.stderr}")

    except Exception as e:
        print(f"Error running test: {e}")
        state["tests_passed"] = False

    return state


# Build the graph
def should_continue_coding(state: SessionState) -> bool:
    return not state["tests_passed"] and state["code_attempts"] < 5


def should_regenerate_problem(state: SessionState) -> bool:
    return not state["tests_passed"] and state["code_attempts"] >= 5 and state["problem_attempts"] < 2


def should_end(state: SessionState) -> bool:
    return state["tests_passed"] or state["problem_attempts"] >= 2


# Create the state graph
workflow = StateGraph(SessionState)

# Add nodes
workflow.add_node("problem_agent", problem_agent)
workflow.add_node("code_agent", code_agent)
workflow.add_node("run_tests", run_tests)

# Set entry point
workflow.set_entry_point("problem_agent")

# Add edges
workflow.add_edge("problem_agent", "code_agent")
workflow.add_edge("code_agent", "run_tests")

# Add conditional edges
workflow.add_conditional_edges("run_tests", lambda state: "end" if should_end(state) else ("code_agent" if should_continue_coding(state) else "problem_agent"), {"end": END, "code_agent": "code_agent", "problem_agent": "problem_agent"})

# Compile the graph
app = workflow.compile()


def save_graph_visualization() -> None:
    """Save the workflow graph as PNG, Mermaid diagram and ASCII representation."""

    # Get the graph object
    graph = app.get_graph(xray=True)

    # Save PNG directly using LangGraph's built-in method
    png_path = os.path.join(os.path.dirname(__file__), "workflow_graph.png")
    png_data = graph.draw_mermaid_png()

    with open(png_path, "wb") as f:
        f.write(png_data)

    print(f"✅ PNG diagram saved to: {png_path}")


# Example usage
if __name__ == "__main__":
    # Note: Environment variables are loaded from .env file

    # # Save the graph visualization first
    # print("📊 Saving workflow graph visualization...")
    # save_graph_visualization()

    initial_state = create_initial_state(user_prompt="Generate a math problem for data engeineering intern", topic=ProblemTopic.STRINGS, difficulty=DifficultyLevel.EASY)

    try:
        # Run the workflow
        final_state = app.invoke(initial_state)

        print(f"Tests passed: {final_state['tests_passed']}")
        print(f"Code attempts: {final_state['code_attempts']}")
        print(f"Problem attempts: {final_state['problem_attempts']}")
        print(f"Final code:\n{final_state['code']}")

        if final_state["problem"]:
            print(f"\nProblem description: {final_state['problem'].description}")
            print(f"Tests: {final_state['problem'].tests}")

    except Exception as e:
        print(f"Error running workflow: {e}")
        print("Make sure you have set the required environment variables:")
        print("- MISTRAL_API_KEY")
        print("- E2B_API_KEY")
