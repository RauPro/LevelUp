import os

from dotenv import load_dotenv
from mlflow.metrics.genai import EvaluationExample, make_genai_metric

load_dotenv()

MODEL_URI = "Qwen/QwQ-32B"

# This assumes you have set your OpenAI-compatible API key and base URL
# as environment variables, which is good practice.
# os.environ["OPENAI_API_KEY"] = "your-api-key"
# os.environ["OPENAI_API_BASE"] = "your-endpoint-url"

API_KEY = os.getenv("NEBIUS_API_KEY")
difficulty_accuracy_metric = make_genai_metric(
    name="difficulty_accuracy",
    definition="Evaluates whether the generated problem matches the requested difficulty level.",
    grading_prompt=(
        "You are an expert judge evaluating programming problems. Your task is to assess if the generated problem's difficulty "
        "matches the level requested in the original query. Use the scoring rubric below and provide a single integer score from 1 to 5.\n\n"
        "Scoring Rubric:\n"
        "- Score 1: Problem difficulty is completely mismatched (e.g., trivial when Hard was requested).\n"
        "- Score 2: Problem difficulty is significantly off from the requested level.\n"
        "- Score 3: Problem difficulty is somewhat close but noticeably easier or harder than requested.\n"
        "- Score 4: Problem difficulty is close to the requested level with minor variance.\n"
        "- Score 5: Problem difficulty perfectly matches the requested level.\n\n"
        "Evaluate the following:\n"
        "Original query: {inputs}\n"
        "Generated problem: {predictions}\n\n"
        "Your response MUST be a single integer from 1 to 5 and nothing else. Do not provide any explanation or introductory text. For example, if the score is 4, your response must be exactly '4'."
    ),
    examples=[
        EvaluationExample(
            input="Generate an Easy Dynamic Programming problem suitable for a junior Software Engineer interview.",
            output=(
                "Problem Statement: The Enchanted Garden\n\n#### Description\n"
                "In a mystical land, there exists an enchanted garden filled with flowers that bloom in various colors. "
                "Each flower has a unique energy value associated with it... (rest of your example output)"
            ),
            score=5,
            justification="The problem is perfectly appropriate for an Easy difficulty level, using a simple DP approach with small constraints and clear explanation.",
        )
    ],
    version="v1",
    model=f"openai:/{MODEL_URI}",
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
    parameters={"temperature": 0.0},
)


topic_relevance_metric = make_genai_metric(
    name="topic_relevance",
    definition="Measures how well the generated problem matches the requested topic/algorithm in the query.",
    grading_prompt=(
        "You are an expert judge evaluating programming problems. Your task is to assess how well the generated problem aligns "
        "with the algorithm or topic requested in the original query. Use the scoring rubric below and provide a single integer score from 1 to 5.\n\n"
        "Scoring Rubric:\n"
        "- Score 1: Problem is completely unrelated to the requested topic.\n"
        "- Score 2: Problem has minimal relation to the requested topic.\n"
        "- Score 3: Problem is somewhat related to the topic but misses key aspects.\n"
        "- Score 4: Problem is clearly related to the requested topic with minor misalignments.\n"
        "- Score 5: Problem perfectly matches the requested topic/algorithm.\n\n"
        "Evaluate the following:\n"
        "Original query: {inputs}\n"
        "Generated problem: {predictions}\n\n"
        "Your response MUST be a single integer from 1 to 5 and nothing else. Do not provide any explanation or introductory text. For example, if the score is 5, your response must be exactly '5'."
    ),
    examples=[
        EvaluationExample(
            input="Generate an Easy Dynamic Programming problem suitable for a junior Software Engineer interview.",
            output=(
                "Problem Statement: The Enchanted Garden\n\n#### Description\n"
                "In a mystical land, there exists an enchanted garden filled with flowers that bloom in various colors. "
                "Each flower has a unique energy value associated with it... (rest of your example output)"
            ),
            score=5,
            # Corrected justification to match the input
            justification="The problem is a classic Dynamic Programming task, perfectly matching the request.",
        )
    ],
    version="v1",
    model=f"openai:/{MODEL_URI}",
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
    parameters={"temperature": 0.0},
)