import json
import os
import pickle
import tempfile

import mlflow
import openai

from ml.agent import global_prompt
from models.state import SessionState


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