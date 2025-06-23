# ml/rag_pipeline.py
import os
import json

from dotenv import load_dotenv

from app.api.schemas import ProblemRequest, ProblemResponse
from ml.embedding_generator import problem_collection
import openai

# The generate_llm_prompt function remains the same as before.
def generate_llm_prompt(topic: str, difficulty: str, retrieved_problems: list) -> str:
    """Creates a detailed prompt for the LLM."""
    prompt = f"You are an expert problem setter for a technical interview platform.\n"
    prompt += f"Your task is to create a new, unique programming problem on the topic of '{topic.title()}' with a '{difficulty.upper()}' difficulty level.\n\n"
    prompt += "To help you, here are some examples of existing problems on the same topic. Do NOT copy them directly. Use them as inspiration for style, structure, and difficulty.\n\n"
    prompt += "--- EXAMPLES ---\n"
    for i, prob in enumerate(retrieved_problems):
        prompt += f"Example {i + 1}:\n"
        prompt += f"Title: {prob['metadatas']['name']}\n"
        prompt += f"Description: {prob['documents']}\n\n"
    prompt += "--- END OF EXAMPLES ---\n\n"
    prompt += (
        "Now, generate a brand new problem. IMPORTANT: You must respond with only the JSON object for the problem, "
        "following this exact structure:\n"
        "{\n"
        "  'id': '...',  // Generate a unique identifier\n"
        "  'title': '...',  // A concise title for the problem\n"
        "  'description': '...',  // Detailed problem description\n"
        "  'constraints': ['...', '...'],  // List of constraints as strings\n"
        "  'examples': [\n"
        "    {\n"
        "      'input': '...',\n"
        "      'output': '...',\n"
        "      'explanation': '...'  // Optional field\n"
        "    },\n"
        "    // You can include multiple examples\n"
        "  ],\n"
        f"  'difficulty': '{difficulty.lower()}',\n"
        f"  'topic': '{topic.lower()}',\n"
        "  'python_solution': '...'  // Complete, runnable Python function that solves the problem with example calls\n"
        "}\n"
        "Make sure to include at least one example with input, output, and explanation. Make sure Python code is runnable and solves the problem.\n"
    )

    return prompt


async def generate_new_problem(request: ProblemRequest) -> ProblemResponse:
    """
    The main RAG pipeline function using the OpenAI API.
    """

    query_text = f"{request.topic.value} {request.difficulty.value} {request.user_prompt}"

    retrieved = problem_collection.query(
        query_texts=[query_text],
        n_results=3,
    )

    retrieved_problems = [
        {"documents": doc, "metadatas": meta}
        for doc, meta in zip(retrieved["documents"][0], retrieved["metadatas"][0])
    ]

    prompt = generate_llm_prompt(request.topic.value, request.difficulty.value, retrieved_problems)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
    model_name = "gpt-4o"

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = openai.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"}
    )
    print(response.choices[0].message.content)
    generated_content = response.choices[0].message.content
    import ast
    max_retries = 10
    retries = 0
    success = False

    while retries < max_retries and not success:
        try:
            new_problem = ast.literal_eval(generated_content)
            success = True
        except (SyntaxError, ValueError) as e:
            print(f"Parsing error (attempt {retries + 1}/{max_retries}): {e}")
            retries += 1

            if retries < max_retries:
                retry_messages = messages.copy()
                chat_response = openai.chat.completions.create(
        model=model_name,
        messages=retry_messages,
        response_format={"type": "json_object"}
    )
                generated_content = chat_response.choices[0].message.content
            else:
                print(f"Failed to parse response after {max_retries} attempts")
                new_problem = {
                    "title": "Error Generating Problem",
                    "description": "There was an error generating the problem. Please try again."
                }

    if not success:
        new_problem = {
            "title": "Error Generating Problem",
            "description": "There was an error generating the problem. Please try again."
        }
    return new_problem
