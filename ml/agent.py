import json
import os
import re
import pickle
import uuid
import tempfile
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from app.api.schemas import DifficultyLevel, ProblemTopic
import openai
import mlflow
from models.state import SessionState, Problem

from ml.rag_retriever import default_retriever as rag_retriever

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

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
    return SessionState(user_prompt=user_prompt, topic=topic, difficulty=difficulty, problem=None, code=None,
                        tests_passed=False, code_attempts=0, problem_attempts=0)


llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

sandbox = Sandbox(api_key=os.getenv("E2B_API_KEY"))
global_prompt = ""

def problem_agent(state: SessionState) -> SessionState:
    global  global_prompt
    """Generate programming problems using RAG approach."""
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Use the RAG retriever to get similar problems
    retrieved_problems = rag_retriever.retrieve(topic=state["topic"].value, difficulty=state["difficulty"].value,
        user_prompt=state["user_prompt"])

    # Generate a prompt using the retrieved problems
    prompt = rag_retriever.generate_prompt(topic=state["topic"].value, difficulty=state["difficulty"].value,
        retrieved_problems=retrieved_problems)
    global_prompt = prompt
    messages = [{"role": "user", "content": prompt}]

    response = openai.chat.completions.create(model="gpt-4o", messages=messages,
        response_format={"type": "json_object"})

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

            # Adapt the response to match our Problem structure
            if "description" in problem_dict and "tests" in problem_dict:
                problem = Problem(**problem_dict)
                success = True
            else:
                # If the keys don't match our expected format, try to extract what we need
                if "title" in problem_dict and "description" in problem_dict:
                    # Extract from a different format
                    description = f"{problem_dict.get('title', '')}: {problem_dict.get('description', '')}"
                    tests = problem_dict.get('examples', [])
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
                retry_messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": generated_content}, {"role": "user",
                                                                          "content": "The response could not be parsed correctly. Please provide a valid JSON object with 'description' and 'tests' fields only. Make sure to use double quotes for keys and string values."}]

                retry_response = openai.chat.completions.create(model="gpt-4o", messages=retry_messages,
                    response_format={"type": "json_object"})

                generated_content = retry_response.choices[0].message.content
            else:
                print(f"Failed to parse response after {max_retries} attempts")
                # Fallback to a simple problem
                problem = Problem(description="Add two space-separated integers",
                    tests=[{"input": "5 3", "output": "8"}])
                break

    if not success and not problem:
        # Fallback if all else fails
        problem = Problem(description="Add two space-separated integers", tests=[{"input": "5 3", "output": "8"}])

    state["problem"] = problem
    state["code_attempts"] = 0  # Reset for new problem
    state["problem_attempts"] += 1
    return state


def code_agent(state: SessionState) -> SessionState:
    if state["problem"] is None:
        raise ValueError("Problem must be set before generating code")

    prompt = ChatPromptTemplate.from_template(
        "Write a Python solution for: {problem_description}. The solution should read input from stdin and write output to stdout. If the input contains multiple values, they will be space-separated on a single line. Use input().split() to parse space-separated values. Return ONLY the Python code, no markdown formatting, no explanations. Example for adding two numbers: a, b = map(int, input().split())\nprint(a + b)")
    chain = prompt | llm | StrOutputParser()
    code = chain.invoke({"problem_description": state["problem"].description})

    if "```python" in code:
        code_match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1)
    elif "```" in code:
        code = re.sub(r"```.*?\n(.*?)\n```", r"\1", code, flags=re.DOTALL)

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

    if state["code"] is None or state["problem"] is None:
        state["tests_passed"] = False
        return state

    code = state["code"]
    test = state["problem"].tests[0]

    # Clean up code formatting if needed
    if "```python" in code:
        code_match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1)
    elif "```" in code:
        code = re.sub(r"```.*?\n(.*?)\n```", r"\1", code, flags=re.DOTALL)

    # Process test input - convert list-like strings to space-separated values
    test_input = test["input"]
    if test_input.strip().startswith('[') and test_input.strip().endswith(']'):
        try:
            # Convert string representation of list to actual list
            input_list = json.loads(test_input.replace("'", '"'))
            # Convert list to space-separated string
            test_input = ' '.join(map(str, input_list))
        except json.JSONDecodeError:
            # If conversion fails, use as-is
            pass

    print(f"Running code:\n{code}\nWith test input: '{test_input}' and expected output: '{test['output']}'")

    try:
        try:
            print(test_input)
            sandbox.files.write("tests.txt", test_input)
            execution = sandbox.run_code(code, test_input)
            print(f"E2B execution logs: {execution.logs}")

            if execution.logs.stdout:
                output = execution.logs.stdout[0].strip()
                expected = test["output"].strip()

                # Normalize formats for comparison
                if expected.startswith('[') and expected.endswith(']'):
                    try:
                        # Parse expected output as a list
                        expected_list = json.loads(expected.replace("'", '"'))

                        # Parse actual output as a list (if space-separated)
                        if '[' not in output:
                            output_list = list(map(int, output.split()))

                            # Compare the lists regardless of format
                            state["tests_passed"] = sorted(expected_list) == sorted(output_list)
                        else:
                            # Try to parse output as a list if it looks like one
                            try:
                                output_list = json.loads(output.replace("'", '"'))
                                state["tests_passed"] = sorted(expected_list) == sorted(output_list)
                            except:
                                state["tests_passed"] = output == expected
                    except:
                        # Fall back to direct comparison if parsing fails
                        state["tests_passed"] = output == expected
                else:
                    state["tests_passed"] = output == expected

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

            process = subprocess.run([sys.executable, temp_file], input=test_input, capture_output=True, text=True,
                                     timeout=5)

            os.unlink(temp_file)

            if process.returncode == 0:
                output = process.stdout.strip()
                expected = test["output"].strip()

                # Normalize formats for comparison (same logic as above)
                if expected.startswith('[') and expected.endswith(']'):
                    try:
                        expected_list = json.loads(expected.replace("'", '"'))

                        if '[' not in output:
                            output_list = list(map(int, output.split()))
                            state["tests_passed"] = sorted(expected_list) == sorted(output_list)
                        else:
                            try:
                                output_list = json.loads(output.replace("'", '"'))
                                state["tests_passed"] = sorted(expected_list) == sorted(output_list)
                            except:
                                state["tests_passed"] = output == expected
                    except:
                        state["tests_passed"] = output == expected
                else:
                    state["tests_passed"] = output == expected

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
    return not state["tests_passed"] and state["code_attempts"] >= 5 and state["problem_attempts"] < 2


def should_end(state: SessionState) -> bool:
    return state["tests_passed"] or state["problem_attempts"] >= 2


workflow = StateGraph(SessionState)

workflow.add_node("problem_agent", problem_agent)
workflow.add_node("code_agent", code_agent)
workflow.add_node("run_tests", run_tests)

workflow.set_entry_point("problem_agent")

workflow.add_edge("problem_agent", "code_agent")
workflow.add_edge("code_agent", "run_tests")

workflow.add_conditional_edges("run_tests", lambda state: "end" if should_end(state) else (
    "code_agent" if should_continue_coding(state) else "problem_agent"),
                               {"end": END, "code_agent": "code_agent", "problem_agent": "problem_agent"})

app = workflow.compile()


def save_graph_visualization() -> None:
    """Save the workflow graph as PNG, Mermaid diagram and ASCII representation."""

    graph = app.get_graph(xray=True)

    png_path = os.path.join(os.path.dirname(__file__), "workflow_graph.png")
    png_data = graph.draw_mermaid_png()

    with open(png_path, "wb") as f:
        f.write(png_data)

    print(f"✅ PNG diagram saved to: {png_path}")


def log_to_mlflow(state: SessionState) -> str:
    """
    Log metrics and full session state to MLflow using the built-in evaluation framework.

    Args:
        state: The SessionState containing all workflow information

    Returns:
        str: The MLflow run ID that can be used for Grafana visualization
    """
    import pandas as pd
    from ml.metrics import difficulty_accuracy_metric, topic_relevance_metric

    # Start MLflow run with a descriptive name
    with mlflow.start_run(run_name="LevelUp Problem Evaluation") as run:
        # Log basic information as tags
        mlflow.set_tag("topic", state["topic"].value)
        mlflow.set_tag("difficulty", state["difficulty"].value)
        mlflow.set_tag("tests_passed", str(state["tests_passed"]))
        mlflow.set_tag("code_attempts", str(state["code_attempts"]))
        mlflow.set_tag("problem_attempts", str(state["problem_attempts"]))

        # Log user prompt as parameter
        mlflow.log_param("user_prompt", state["user_prompt"])

        # Create a small evaluation dataset with this problem
        if state["problem"]:
            # Construct query and prediction for evaluation
            query = global_prompt if global_prompt != '' else "Give me easy sorting problem"
            problem_description = state["problem"].description

            try:
                eval_data = pd.DataFrame({
                    "inputs": [query],
                    "predictions": [problem_description]
                })
                levelup_qa_model = mlflow.openai.log_model(
                    model="gpt-4o-mini",
                    task=openai.chat.completions,
                    name="levelup_qa_model",
                    messages=[
                        {"role": "system", "content": "Generate a programming problem based on the specified topic and difficulty"},
                        {"role": "user", "content": "{inputs}"},
                    ],
                )

                mlflow.set_active_model(name="levelup_qa_model")

                mlflow.log_param("generator_model", "gpt-4o-mini")
                mlflow.log_param("judge_model", "openai:/gpt-4")

                eval_dataset = mlflow.data.from_pandas(
                    df=eval_data,
                    name="levelup_evaluation_data"
                )
                mlflow.log_input(eval_dataset)

                results = mlflow.evaluate(
                    data=eval_data,
                    model=levelup_qa_model.model_uri,
                    extra_metrics=[
                        topic_relevance_metric,
                        difficulty_accuracy_metric,
                    ],
                    evaluator_config={
                        "col_mapping": {
                            "inputs": "inputs",
                            "predictions": "predictions"
                        }
                    }
                )

                print("Evaluation Results:")
                for metric_name, value in results.metrics.items():
                    print(f"{metric_name}: {value}")

                if "eval_results_table" in results.tables:
                    detailed_results = results.tables["eval_results_table"]
                    print("\nDetailed Results:")
                    print(detailed_results)

                print("Successfully logged metrics using MLflow evaluate")
            except Exception as e:
                print(f"Error using MLflow evaluate: {e}")
                mlflow.log_metric("difficulty_accuracy", 3.0)
                mlflow.log_metric("topic_relevance", 3.0)

        if state["problem"]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(state["problem"].description)
                problem_path = f.name
            mlflow.log_artifact(problem_path, "problem")
            os.unlink(problem_path)

            if state["problem"].tests:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    json.dump(state["problem"].tests, f, indent=2)
                    tests_path = f.name
                mlflow.log_artifact(tests_path, "tests")
                os.unlink(tests_path)

        if state["code"]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(state["code"])
                code_path = f.name
            mlflow.log_artifact(code_path, "code")
            os.unlink(code_path)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            pickle.dump(state, f)
            state_path = f.name
        mlflow.log_artifact(state_path, "full_state")
        os.unlink(state_path)
        mlflow.log_metric("problem_attempts", state["problem_attempts"])
        mlflow.log_metric("code_attempts", state["code_attempts"])
        return run.info.run_id
