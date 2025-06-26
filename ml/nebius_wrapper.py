import os

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv
from mlflow.pyfunc import PythonModelContext

load_dotenv()


class NebiusWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow wrapper for Nebius API to enable evaluation with custom API key.
    """

    def __init__(self):
        self.api_key = os.getenv("NEBIUS_API_KEY")
        self.model_uri = "Qwen/QwQ-32B"
        self.base_url = "https://api.studio.nebius.ai/v1/chat/completions"

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame) -> list[str]:
        """
        Make predictions using Nebius API.

        Args:
            context: MLflow context (not used)
            model_input: Pandas DataFrame with 'inputs' column

        Returns:
            List of generated responses
        """
        if isinstance(model_input, pd.DataFrame):
            inputs = model_input["inputs"].tolist()
        else:
            inputs = [model_input] if isinstance(model_input, str) else model_input

        predictions = []

        for input_text in inputs:
            try:
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

                payload = {"model": self.model_uri, "messages": [{"role": "system", "content": "Generate a programming problem based on the specified topic and difficulty"}, {"role": "user", "content": input_text}], "temperature": 0.0, "max_tokens": 4000}

                response = requests.post(self.base_url, headers=headers, json=payload, timeout=70)

                if response.status_code == 200:
                    result = response.json()
                    prediction = result["choices"][0]["message"]["content"]
                    predictions.append(prediction)
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    predictions.append(f"Error: Failed to generate response (Status: {response.status_code})")

            except Exception as e:
                print(f"Error calling Nebius API: {e}")
                predictions.append(f"Error: {str(e)}")

        return predictions
