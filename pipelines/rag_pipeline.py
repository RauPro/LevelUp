# ml/rag_pipeline.py
import os
from typing import Any

from dotenv import load_dotenv

# Import the official Mistral AI client
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage  # type: ignore

from app.api.schemas import ProblemRequest
from ml.embedding_generator import problem_collection

load_dotenv()


# The generate_llm_prompt function remains the same as before.
def generate_llm_prompt(topic: str, difficulty: str, retrieved_problems: list[dict[str, Any]]) -> str:
    """Creates a detailed prompt for the LLM."""
    prompt = "You are an expert problem setter for a technical interview platform.\n"
    prompt += f"Your task is to create a new, unique programming problem on the topic of '{topic.title()}' with a '{difficulty.upper()}' difficulty level.\n\n"
    prompt += "To help you, here are some examples of existing problems on the same topic. Do NOT copy them directly. Use them as inspiration for style, structure, and difficulty.\n\n"
    prompt += "--- EXAMPLES ---\n"
    for i, prob in enumerate(retrieved_problems):
        prompt += f"Example {i + 1}:\n"
        prompt += f"Title: {prob['metadatas']['name']}\n"
        prompt += f"Description: {prob['documents']}\n\n"
    prompt += "--- END OF EXAMPLES ---\n\n"
    prompt += "Now, generate a brand new problem. IMPORTANT: You must respond with only the JSON object for the problem, following this exact structure: { 'title': '...', 'description': '...', 'constraints': ['...'], 'example': { 'input': '...', 'output': '...', 'explanation': '...' }, 'python_solution': '...' }\n"

    return prompt


async def generate_new_problem(request: ProblemRequest) -> str:
    """
    The main RAG pipeline function using the Mistral AI client.
    """
    # 1. Retrieval part (this stays the same)
    query_text = f"{request.topic.value} {request.difficulty.value} {request.user_prompt}"

    retrieved = problem_collection.query(
        query_texts=[query_text],
        n_results=3,
    )

    # Handle None cases for mypy
    if retrieved["documents"] is None or retrieved["metadatas"] is None:
        retrieved_problems = []
    else:
        retrieved_problems = [{"documents": doc, "metadatas": meta} for doc, meta in zip(retrieved["documents"][0], retrieved["metadatas"][0])]

    # 2. Augmented Prompt Generation (this stays the same)
    prompt = generate_llm_prompt(request.topic.value, request.difficulty.value, retrieved_problems)

    # 3. Generation part (this is updated for mistralai client)
    api_key = os.getenv("MISTRAL_API_KEY")
    client = MistralClient(api_key=api_key)

    # Use the 'mistral-tiny' model
    model_name = "mistral-tiny"

    # Format the message using ChatMessage
    messages = [ChatMessage(role="user", content=prompt)]

    # Call the Mistral API
    # You can use response_format to enforce JSON output with newer Mistral models
    chat_response = client.chat(  # type: ignore
        model=model_name,
        messages=messages,
        # response_format={"type": "json_object"} # Uncomment if using a model that supports this
    )
    generated_content = chat_response.choices[0].message.content
    # new_problem = json.loads(generated_content)
    return generated_content or ""
