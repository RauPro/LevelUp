import os

from dotenv import load_dotenv
from mlflow.metrics.genai import EvaluationExample, make_genai_metric

load_dotenv()

MODEL_URI = "Qwen/QwQ-32B"

API_KEY = os.getenv("NEBIUS_API_KEY")

difficulty_accuracy_metric = make_genai_metric(
    name="difficulty_accuracy",
    definition="Evaluates whether the generated problem matches the requested difficulty level.",
    grading_prompt=(
        "Assess if the difficulty level of the generated problem matches what was requested in the query.\n\n"
        "Score on a scale of 1-5:\n"
        "- Score 1: Problem difficulty is completely mismatched (e.g., trivial when Hard was requested).\n"
        "- Score 2: Problem difficulty is significantly off from the requested level.\n"
        "- Score 3: Problem difficulty is somewhat close but noticeably easier or harder than requested.\n"
        "- Score 4: Problem difficulty is close to the requested level with minor variance.\n"
        "- Score 5: Problem difficulty perfectly matches the requested level.\n\n"
        "Original query: {inputs}\n"
        "Generated problem: {predictions}\n"
        "IMPORTANT: RETURN ONLY THE SCORE AS AN INTEGER FROM 1 TO 5, WITHOUT ANY ADDITIONAL TEXT OR EXPLANATION.\n"
    ),
examples=[
        EvaluationExample(
            input="Generate an Easy Dynamic Programming problem suitable for a junior Software Engineer interview.",
            output=(
                "Problem Statement: The Enchanted Garden\n\n#### Description\n"
                "In a mystical land, there exists an enchanted garden filled with flowers that bloom in various colors. "
                "Each flower has a unique energy value associated with it, and the garden has a specific layout represented as a 2D grid. "
                "The garden's rows and columns are numbered from 0 to n-1 and 0 to m-1, respectively.\n\n"
                "You are tasked with helping a wizard collect the maximum energy from the garden while adhering to specific movement rules. "
                "The wizard can start at any flower in the first row and can move to the adjacent flowers in the next row "
                "(i.e., directly below, diagonally left, or diagonally right). "
                "The objective is to find the maximum energy the wizard can collect by moving from the top to the bottom of the garden grid.\n\n"
                "Write a function `max_energy(garden: List[List[int]]) -> int` that takes a 2D list representing the garden "
                "and returns the maximum energy the wizard can collect.\n\n"
                "#### Constraints\n"
                "- The garden grid will have dimensions `n x m` (1 ≤ n, m ≤ 100).\n"
                "- Each energy value in the grid will be an integer in the range of -1000 to 1000.\n\n"
                "#### Sample Input\n```python\ngarden = [\n    [1, 2, 3],\n    [4, 5, 6],\n    [7, 8, 9]\n]\n```\n\n"
                "#### Sample Output\n```python\n15\n```\n\n"
                "#### Explanation\n"
                "The wizard can start at flower `3` (row 0, column 2), move to flower `6` (row 1, column 2), "
                "and finally to flower `9` (row 2, column 2), collecting energy values `3 + 6 + 9 = 18`. "
                "A more optimal path would be starting at flower `1` (row 0, column 0), moving to flower `5` (row 1, column 1), "
                "and then to flower `9` (row 2, column 2), yielding a total energy of `1 + 5 + 9 = 15`.\n\n"
                "The goal of the problem is to implement the function efficiently to handle the maximum constraints "
                "while ensuring the correct energy accumulation according to the specified movement rules."
            ),
            score=5,
            justification="The problem is perfectly appropriate for an Easy difficulty level, using a simple DP approach with small constraints and clear explanation.",
        )],
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
        "Evaluate how well the generated problem aligns with the algorithm/topic requested in the original query.\n\n"
        "Score on a scale of 1-5:\n"
        "- Score 1: Problem is completely unrelated to the requested topic.\n"
        "- Score 2: Problem has minimal relation to the requested topic.\n"
        "- Score 3: Problem is somewhat related to the topic but misses key aspects.\n"
        "- Score 4: Problem is clearly related to the requested topic with minor misalignments.\n"
        "- Score 5: Problem perfectly matches the requested topic/algorithm.\n\n"
        "Original query: {inputs}\n"
        "Generated problem: {predictions}\n"
        "IMPORTANT: RETURN ONLY THE SCORE AS AN INTEGER FROM 1 TO 5, WITHOUT ANY ADDITIONAL TEXT OR EXPLANATION.\n"
    ),
    examples=[
        EvaluationExample(
            input="Generate an Easy Dynamic Programming problem suitable for a junior Software Engineer interview.",
            output=(
                "Problem Statement: The Enchanted Garden\n\n#### Description\n"
                "In a mystical land, there exists an enchanted garden filled with flowers that bloom in various colors. "
                "Each flower has a unique energy value associated with it, and the garden has a specific layout represented as a 2D grid. "
                "The garden's rows and columns are numbered from 0 to n-1 and 0 to m-1, respectively.\n\n"
                "You are tasked with helping a wizard collect the maximum energy from the garden while adhering to specific movement rules. "
                "The wizard can start at any flower in the first row and can move to the adjacent flowers in the next row "
                "(i.e., directly below, diagonally left, or diagonally right). "
                "The objective is to find the maximum energy the wizard can collect by moving from the top to the bottom of the garden grid.\n\n"
                "Write a function `max_energy(garden: List[List[int]]) -> int` that takes a 2D list representing the garden "
                "and returns the maximum energy the wizard can collect.\n\n"
                "#### Constraints\n"
                "- The garden grid will have dimensions `n x m` (1 ≤ n, m ≤ 100).\n"
                "- Each energy value in the grid will be an integer in the range of -1000 to 1000.\n\n"
                "#### Sample Input\n```python\ngarden = [\n    [1, 2, 3],\n    [4, 5, 6],\n    [7, 8, 9]\n]\n```\n\n"
                "#### Sample Output\n```python\n15\n```\n\n"
                "#### Explanation\n"
                "The wizard can start at flower `3` (row 0, column 2), move to flower `6` (row 1, column 2), "
                "and finally to flower `9` (row 2, column 2), collecting energy values `3 + 6 + 9 = 18`. "
                "A more optimal path would be starting at flower `1` (row 0, column 0), moving to flower `5` (row 1, column 1), "
                "and then to flower `9` (row 2, column 2), yielding a total energy of `1 + 5 + 9 = 15`.\n\n"
                "The goal of the problem is to implement the function efficiently to handle the maximum constraints "
                "while ensuring the correct energy accumulation according to the specified movement rules."
            ),
            score=5,
            justification="The problem directly implements binary search as requested in the query.",
        )
    ],
    version="v1",
    model=f"openai:/{MODEL_URI}",
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
    parameters={"temperature": 0.0},
)
