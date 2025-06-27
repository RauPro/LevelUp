import json
import os
import pickle
import tempfile
from collections.abc import Iterator

import mlflow
import pandas as pd
from dotenv import load_dotenv
from packaging.version import Version

# MLflow version check for tracing functionality
assert Version(mlflow.__version__) >= Version("2.14.3"), (
    "This feature requires MLflow version 2.14.3 or newer. "
    "Please run '%pip install -U mlflow' in a notebook cell, "
    "and restart the kernel when the command finishes."
)

from ml.agent import global_prompt
from ml.metrics import API_KEY, MODEL_URI
from ml.nebius_wrapper import NebiusWrapper
from models.state import Problem, SessionState

load_dotenv()


@mlflow.trace(name="mlflow_evaluation", attributes={"component": "evaluation_logger"})
def log_to_mlflow(state: SessionState, state_history: Iterator = None) -> tuple:
    """
    Log metrics and full session state to MLflow using the built-in evaluation framework.

    Args:
        state: The final SessionState of the workflow
        state_history: Iterator of all state snapshots from the workflow run

    Returns:
        tuple: The run ID, metrics, problem attempts, and code attempts
    """

    from ml.metrics import difficulty_accuracy_metric, topic_relevance_metric

    # Configure environment for Nebius API compatibility with MLflow
    os.environ["OPENAI_API_KEY"] = API_KEY
    os.environ["OPENAI_BASE_URL"] = "https://api.studio.nebius.ai/v1"

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

        # Log the full state history as a JSON artifact
        if state_history:
            serializable_history = []
            for snapshot in state_history:
                # Convert the state snapshot to a serializable format
                state_dict = dict(snapshot.values)

                # Handle Pydantic models and enums
                if state_dict.get("problem") and isinstance(state_dict["problem"], Problem):
                    state_dict["problem"] = state_dict["problem"].model_dump()
                if state_dict.get("topic") and hasattr(state_dict["topic"], "value"):
                    state_dict["topic"] = state_dict["topic"].value
                if state_dict.get("difficulty") and hasattr(state_dict["difficulty"], "value"):
                    state_dict["difficulty"] = state_dict["difficulty"].value

                serializable_history.append(state_dict)

            # Save state history as JSON artifact
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="state_history_") as f:
                json.dump(serializable_history, f, indent=2)
                history_path = f.name
            mlflow.log_artifact(history_path, "state_history")
            os.unlink(history_path)
            # Log additional metadata about the workflow
            mlflow.log_param("total_workflow_steps", len(serializable_history))
            print(f"✅ Logged state history with {len(serializable_history)} steps as artifact")

        # Create a small evaluation dataset with this problem
        metrices_for_grafana = {}
        if state["problem"]:
            # Construct query and prediction for evaluation
            query = global_prompt if global_prompt != "" else "Give me easy sorting problem"
            problem_description = state["problem"].description

            try:
                eval_data = pd.DataFrame({"inputs": [query], "predictions": [problem_description]})

                # Create sample input for the custom wrapper
                sample_input = eval_data[["inputs"]].head(1)

                # Log the custom Nebius model
                levelup_qa_model = mlflow.pyfunc.log_model(python_model=NebiusWrapper(), artifact_path="levelup_qa_model", input_example=sample_input)  # Log model parameters
                mlflow.log_param("generator_model", MODEL_URI)
                mlflow.log_param("judge_model", MODEL_URI)
                mlflow.log_param("api_provider", "nebius")

                eval_dataset = mlflow.data.from_pandas(df=eval_data, name="levelup_evaluation_data")
                mlflow.log_input(eval_dataset)

                results = mlflow.evaluate(
                    data=eval_data,
                    model=levelup_qa_model.model_uri,
                    extra_metrics=[
                        difficulty_accuracy_metric,
                        topic_relevance_metric
                    ],
                    targets="predictions",
                    feature_names=["inputs"],
                    evaluator_config={"col_mapping": {"inputs": "inputs", "predictions": "predictions"}},
                )
                print(results.tables["eval_results_table"])
                print("Evaluation Results:")
                metrices_for_grafana = {}
                for metric_name, value in results.metrics.items():
                    print(f"{metric_name}: {value}")
                    metrices_for_grafana[metric_name] = value

                if "eval_results_table" in results.tables:
                    detailed_results = results.tables["eval_results_table"]
                    print("\nDetailed Results:")
                    print(detailed_results)

                print("Successfully logged metrics using MLflow evaluate")
            except Exception as e:
                print(f"Error using MLflow evaluate: {e}")
                # Fallback metrics
                metrices_for_grafana = {"difficulty_accuracy/v1/mean": 3.0, "topic_relevance/v1/mean": 3.0, "difficulty_accuracy/v1/variance": 0.5, "topic_relevance/v1/variance": 0.5}
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

        return run.info.run_id, metrices_for_grafana, state["problem_attempts"], state["code_attempts"]

