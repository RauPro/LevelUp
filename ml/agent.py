import json
import os
import re
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import openai
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.api.schemas import DifficultyLevel, ProblemTopic

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow

from ml.rag_retriever import default_retriever as rag_retriever
from models.state import Problem, SessionState

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.langchain.autolog()
# Create or set the experiment
EXPERIMENT_NAME = "problem-generation"
try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    print(f"Error setting up MLflow experiment: {e}")
    # Fallback to default experiment
    experiment_id = "0"


def create_initial_state(user_prompt: str, topic: ProblemTopic, difficulty: DifficultyLevel) -> SessionState:
    """Create initial state dictionary."""
    return SessionState(user_prompt=user_prompt, topic=topic, difficulty=difficulty, problem=None, code=None, tests_passed=False, code_attempts=0, problem_attempts=0)


nebius_client = openai.OpenAI(base_url="https://api.studio.nebius.com/v1/", api_key=os.environ.get("NEBIUS_API_KEY"))

sandbox = Sandbox(api_key=os.getenv("E2B_API_KEY"))
global_prompt = ""


@mlflow.trace(name="problem_generation", attributes={"component": "problem_agent"})
def problem_agent(state: SessionState) -> SessionState:
    global global_prompt
    """Generate programming problems using RAG approach."""
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    # Use the RAG retriever to get similar problems
    retrieved_problems = rag_retriever.retrieve(topic=state["topic"].value, difficulty=state["difficulty"].value, user_prompt=state["user_prompt"])

    # Generate a prompt using the retrieved problems
    prompt = rag_retriever.generate_prompt(topic=state["topic"].value, difficulty=state["difficulty"].value, retrieved_problems=retrieved_problems)
    global_prompt = prompt
    messages = [{"role": "user", "content": prompt}]

    response = nebius_client.chat.completions.create(model="Qwen/Qwen2.5-72B-Instruct-fast", messages=messages, response_format={"type": "json_object"})

    generated_content = response.choices[0].message.content
    print(f"Generated content: {generated_content}")

    max_retries = 10
    retries = 0
    success = False
    problem = None

    while retries < max_retries and not success:
        try:
            # Try to parse the JSON response
            problem_dict = json.loads(generated_content)

            # Convert list inputs/outputs to strings
            if "tests" in problem_dict and isinstance(problem_dict["tests"], list):
                for test_case in problem_dict["tests"]:
                    if "input" in test_case and isinstance(test_case["input"], list):
                        test_case["input"] = " ".join(map(str, test_case["input"]))
                    if "output" in test_case and not isinstance(test_case["output"], str):
                        if isinstance(test_case["output"], list):
                            test_case["output"] = " ".join(map(str, test_case["output"]))
                        else:
                            test_case["output"] = str(test_case["output"])

            # Adapt the response to match our Problem structure
            if "description" in problem_dict and "tests" in problem_dict:
                problem = Problem(**problem_dict)
                success = True
            else:
                # If the keys don't match our expected format, try to extract what we need
                if "title" in problem_dict and "description" in problem_dict:
                    # Extract from a different format
                    description = f"{problem_dict.get('title', '')}: {problem_dict.get('description', '')}"
                    tests = problem_dict.get("examples", [])
                    if not tests and "input" in problem_dict and "output" in problem_dict:
                        tests = [{"input": problem_dict["input"], "output": problem_dict["output"]}]

                    problem = Problem(description=description, tests=tests)
                    success = True
                else:
                    raise ValueError(f"Unexpected JSON structure: {problem_dict.keys()}")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Parsing error (attempt {retries + 1}/{max_retries}): {e}")
            retries += 1

            if retries < max_retries:
                # Retry with explicit instructions to fix the format
                retry_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": generated_content},
                    {"role": "user", "content": 'The response could not be parsed correctly. Please provide a valid JSON object with \'description\' and \'tests\' fields only. IMPORTANT: Both \'input\' and \'output\' in each test MUST be strings (not integers or arrays). For example, if the input is a list, format it as a space-separated string like \'1 2 3 4\'. Example format: {"description": "Problem description", "tests": [{"input": "1 2 3", "output": "6"}]}'},
                ]

                retry_response = nebius_client.chat.completions.create(model="Qwen/Qwen2.5-72B-Instruct-fast", messages=retry_messages, response_format={"type": "json_object"})

                generated_content = retry_response.choices[0].message.content
            else:
                print(f"Failed to parse response after {max_retries} attempts")
                # Fallback to a simple problem
                problem = Problem(description="Add two space-separated integers", tests=[{"input": "5 3", "output": "8"}])
                break

    if not success and not problem:
        # Fallback if all else fails
        problem = Problem(description="Add two space-separated integers", tests=[{"input": "5 3", "output": "8"}])

    state["problem"] = problem
    state["code_attempts"] = 0  # Reset for new problem
    state["problem_attempts"] += 1

    return state


@mlflow.trace(name="code_generation", attributes={"component": "code_agent"})
def code_agent(state: SessionState) -> SessionState:
    if state["problem"] is None:
        raise ValueError("Problem must be set before generating code")
    # Initialize LangChain LLM for code generation (using Nebius)
    llm = ChatOpenAI(model_name="Qwen/Qwen2.5-72B-Instruct-fast", openai_api_base="https://api.studio.nebius.com/v1/", openai_api_key=os.environ.get("NEBIUS_API_KEY"))
    prompt = ChatPromptTemplate.from_template("""Write a Python solution for: {problem_description}.

IMPORTANT INPUT INSTRUCTIONS:
1. The solution MUST read input from stdin using input().split() to parse space-separated values
2. Always parse the input as a LIST OF INTEGERS using: list(map(int, input().split()))
3. DO NOT assume the input is a single integer - always handle it as a list of integers
4. Return ONLY the Python code, no markdown formatting, no explanations

Example for processing integers from a single line:
```python
numbers = list(map(int, input().split()))
# process numbers list
print(result)
```""")
    chain = prompt | llm | StrOutputParser()
    code = chain.invoke({"problem_description": state["problem"].description})

    # Enhanced code cleaning to handle markdown and conversational text
    code_match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # Fallback for code that might not be in a block
        lines = code.split("\n")
        # Filter out lines that are not part of the code
        code_lines = [line for line in lines if not line.strip().startswith(("Here is", "This is", "The following"))]
        code = "\n".join(code_lines).strip()

    state["code"] = code
    state["code_attempts"] += 1
    return state


@mlflow.trace(name="test_execution", attributes={"component": "run_tests"})
def run_tests(state: SessionState) -> SessionState:
    """Run tests using E2B sandbox if available, otherwise use local execution."""

    if state["code"] is None or state["problem"] is None:
        state["tests_passed"] = False
        return state

    code = state["code"]
    test = state["problem"].tests[0]

    # Clean up code formatting if needed (moved from code_agent)
    code_match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()

    # Process test input - convert list-like strings to space-separated values
    test_input = test["input"]
    if test_input.strip().startswith("[") and test_input.strip().endswith("]"):
        try:
            # Convert string representation of list to actual list
            input_list = json.loads(test_input.replace("'", '"'))
            # Convert list to space-separated string
            test_input = " ".join(map(str, input_list))
        except json.JSONDecodeError:
            # If conversion fails, use as-is
            pass

    print(f"Running code:\n{code}\nWith test input: '{test_input}' and expected output: '{test['output']}'")

    try:
        try:
            print(test_input)
            # The sandbox.run_code method is deprecated and does not support stdin.
            # Using sandbox.run_python instead.
            execution = sandbox.run_python(code, stdin=test_input)
            print(f"E2B execution logs: {execution.logs}")

            if execution.logs.stdout:
                output = execution.logs.stdout[0].strip()
                expected = test["output"].strip()

                # Normalize boolean representations for comparison
                def normalize_boolean_output(value):
                    if value.lower() in ['true', 'false']:
                        return value.lower()
                    return value

                # Normalize formats for comparison
                if expected.startswith("[") and expected.endswith("]"):
                    try:
                        # Parse expected output as a list
                        expected_list = json.loads(expected.replace("'", '"'))

                        # Parse actual output as a list (if space-separated)
                        if "[" not in output:
                            output_list = list(map(int, output.split()))

                            # Compare the lists regardless of format
                            state["tests_passed"] = sorted(expected_list) == sorted(output_list)
                        else:
                            # Try to parse output as a list if it looks like one
                            try:
                                output_list = json.loads(output.replace("'", '"'))
                                state["tests_passed"] = sorted(expected_list) == sorted(output_list)
                            except (json.JSONDecodeError, ValueError):
                                state["tests_passed"] = output == expected
                    except (json.JSONDecodeError, ValueError):
                        # Fall back to direct comparison if parsing fails
                        state["tests_passed"] = output == expected
                else:
                    # Handle boolean comparisons by normalizing case
                    normalized_output = normalize_boolean_output(output)
                    normalized_expected = normalize_boolean_output(expected)
                    state["tests_passed"] = normalized_output == normalized_expected

                print(f"Expected: '{expected}', Got: '{output}', Passed: {state['tests_passed']}")
            else:
                state["tests_passed"] = False
                print("No stdout output from E2B execution.")

        except Exception as e2b_error:
            print(f"E2B execution failed: {e2b_error}, falling back to local execution...")

            import os
            import subprocess
            import sys
            import tempfile

            print(f"Running test locally with input: '{test_input}'")

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            process = subprocess.run([sys.executable, temp_file], input=test_input, capture_output=True, text=True, timeout=5)

            os.unlink(temp_file)

            if process.returncode == 0:
                output = process.stdout.strip()
                expected = test["output"].strip()

                # Normalize boolean representations for comparison
                def normalize_boolean_output(value):
                    if value.lower() in ['true', 'false']:
                        return value.lower()
                    return value

                # Normalize formats for comparison (same logic as above)
                if expected.startswith("[") and expected.endswith("]"):
                    try:
                        expected_list = json.loads(expected.replace("'", '"'))

                        if "[" not in output:
                            output_list = list(map(int, output.split()))
                            state["tests_passed"] = sorted(expected_list) == sorted(output_list)
                        else:
                            try:
                                output_list = json.loads(output.replace("'", '"'))
                                state["tests_passed"] = sorted(expected_list) == sorted(output_list)
                            except (json.JSONDecodeError, ValueError):
                                state["tests_passed"] = output == expected
                    except (json.JSONDecodeError, ValueError):
                        state["tests_passed"] = output == expected
                else:
                    # Handle boolean comparisons by normalizing case
                    normalized_output = normalize_boolean_output(output)
                    normalized_expected = normalize_boolean_output(expected)
                    state["tests_passed"] = normalized_output == normalized_expected

                print(f"Expected: '{expected}', Got: '{output}', Passed: {state['tests_passed']}")
            else:
                state["tests_passed"] = False
                print(f"Local execution failed: {process.stderr}")

    except Exception as e:
        print(f"Error running test: {e}")
        state["tests_passed"] = False

    return state


def should_continue_coding(state: SessionState) -> bool:
    return not state["tests_passed"] and state["code_attempts"] < 5


def should_regenerate_problem(state: SessionState) -> bool:
    return not state["tests_passed"] and state["code_attempts"] >= 5 and state["problem_attempts"] < 5


def should_end(state: SessionState) -> bool:
    return state["tests_passed"] or state["problem_attempts"] >= 5


workflow = StateGraph(SessionState)

workflow.add_node("problem_agent", problem_agent)
workflow.add_node("code_agent", code_agent)
workflow.add_node("run_tests", run_tests)

workflow.set_entry_point("problem_agent")

workflow.add_edge("problem_agent", "code_agent")
workflow.add_edge("code_agent", "run_tests")

workflow.add_conditional_edges("run_tests", lambda state: "end" if should_end(state) else ("code_agent" if should_continue_coding(state) else "problem_agent"), {"end": END, "code_agent": "code_agent", "problem_agent": "problem_agent"})

# Compile the workflow with a checkpointer
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


def run_and_save_state_history(user_prompt: str, topic: ProblemTopic, difficulty: DifficultyLevel) -> None:
    """Run the workflow and save the state history to a JSON file."""
    # Create initial state
    initial_state = create_initial_state(user_prompt, topic, difficulty)

    # Define config with a unique thread_id
    config = {"configurable": {"thread_id": "run_001"}}

    # Run the workflow
    app.invoke(initial_state, config)

    # Retrieve state history
    state_history = app.get_state_history(config)

    # Convert to list of dictionaries
    history_data = [snapshot.values for snapshot in state_history]

    # Save to JSON
    with open("state_history.json", "w") as f:
        # The state contains Pydantic models and enums, which need to be handled for JSON serialization.
        # We can't directly use json.dump without a custom encoder.
        # A simple way is to convert the list to a JSON string first.
        # This is a workaround if `snapshot.values.dict()` is not available or if the state is not a Pydantic model.
        # Since SessionState is a TypedDict, we manually process it.
        serializable_history = []
        for state_snapshot in history_data:
            # Create a copy to avoid modifying the original state
            state_copy = state_snapshot.copy()
            if "topic" in state_copy and hasattr(state_copy["topic"], "value"):
                state_copy["topic"] = state_copy["topic"].value
            if "difficulty" in state_copy and hasattr(state_copy["difficulty"], "value"):
                state_copy["difficulty"] = state_copy["difficulty"].value
            if "problem" in state_copy and state_copy["problem"] is not None:
                # Assuming 'problem' is a Pydantic model or has a .dict() method
                if hasattr(state_copy["problem"], "model_dump"):
                    state_copy["problem"] = state_copy["problem"].model_dump()
                elif hasattr(state_copy["problem"], "__dict__"):
                    state_copy["problem"] = state_copy["problem"].__dict__
            serializable_history.append(state_copy)

        json.dump(serializable_history, f, indent=2)

    print("State history saved to state_history.json")


# Example usage (commented out to prevent execution during imports):
# run_and_save_state_history(
#     user_prompt="Generate a problem Software Engineer intern can solve in 30 minutes",
#     topic=ProblemTopic.ARRAYS,
#     difficulty=DifficultyLevel.EASY
# )


def save_graph_visualization() -> None:
    """Save the workflow graph as PNG, Mermaid diagram and ASCII representation."""

    graph = app.get_graph(xray=True)

    png_path = os.path.join(os.path.dirname(__file__), "workflow_graph.png")
    png_data = graph.draw_mermaid_png()

    with open(png_path, "wb") as f:
        f.write(png_data)

    print(f"✅ PNG diagram saved to: {png_path}")
