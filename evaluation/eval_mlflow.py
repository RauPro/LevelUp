import json

import mlflow


def log_session_state(final_state: dict) -> None:
    """
    Logs the entire session state into MLflow.

    Args:
        final_state (dict): The final state of the LangGraph workflow.
    """
    # Set the tracking URI to the running MLflow server
    mlflow.set_tracking_uri("http://localhost:5000")

    # Set the experiment name
    mlflow.set_experiment("LangGraph_Workflows")

    # Start a parent run at the application level
    mlflow.start_run(run_name="LangGraph_Workflow_Parent")

    # Inside log_session_state_
    with mlflow.start_run(nested=True):
        # Log parameters (if any key-value pairs are simple types)
        for key, value in final_state.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                print(f"Error logging param {key}: {e}")

        # Log the entire state as an artifact (JSON file)
        try:
            with open("final_state.json", "w") as f:
                json.dump(final_state, f, indent=4)
            mlflow.log_artifact("final_state.json")
        except Exception as e:
            print(f"Error logging artifact: {e}")


# Example state to log into MLflow
example_state = {
    "user_prompt": "Generate a problem for a backend intern",
    "topic": "Dynamic Programming",
    "difficulty": "Easy",
    "problem": {"description": "Find the maximum sum of a subarray", "tests": [{"input": "[1, -2, 3, 4, -1]", "output": "7"}, {"input": "[-1, -2, -3]", "output": "-1"}]},
    "code": "def max_subarray_sum(arr):\n    max_sum = float('-inf')\n    current_sum = 0\n    for num in arr:\n        current_sum = max(num, current_sum + num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum",
    "tests_passed": True,
    "code_attempts": 4,
    "problem_attempts": 1,
}

# Log the example state into MLflow
log_session_state(example_state)
