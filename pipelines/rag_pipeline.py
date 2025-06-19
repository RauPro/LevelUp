# ml/rag_pipeline.py
import os
import json
from app.api.schemas import ProblemRequest
from ml.vector_db import problem_collection

# Import the official Mistral AI client
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


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
    prompt += "Now, generate a brand new problem. IMPORTANT: You must respond with only the JSON object for the problem, following this exact structure: { 'title': '...', 'description': '...', 'constraints': ['...'], 'example': { 'input': '...', 'output': '...', 'explanation': '...' } }\n"

    return prompt


async def generate_new_problem(request: ProblemRequest) -> dict:
    """
    The main RAG pipeline function using the Mistral AI client.
    """
    # 1. Retrieval part (this stays the same)
    query_text = f"{request.topic.value} {request.difficulty.value}"
    if request.keywords:
        query_text += " " + " ".join(request.keywords)

    retrieved = problem_collection.query(
        query_texts=[query_text],
        n_results=3,
    )

    retrieved_problems = [
        {"documents": doc, "metadatas": meta}
        for doc, meta in zip(retrieved["documents"][0], retrieved["metadatas"][0])
    ]

    # 2. Augmented Prompt Generation (this stays the same)
    prompt = generate_llm_prompt(request.topic.value, request.difficulty.value, retrieved_problems)

    # 3. Generation part (this is updated for mistralai client)
    api_key = os.getenv("MISTRAL_API_KEY")
    client = MistralClient(api_key=api_key)

    # Use the 'mistral-tiny' model
    model_name = "mistral-tiny"

    # Format the message using ChatMessage
    messages = [
        ChatMessage(role="user", content=prompt)
    ]

    # Call the Mistral API
    # You can use response_format to enforce JSON output with newer Mistral models
    chat_response = client.chat(
        model=model_name,
        messages=messages,
        # response_format={"type": "json_object"} # Uncomment if using a model that supports this
    )

    generated_content = chat_response.choices[0].message.content
    new_problem = json.loads(generated_content)

    return new_problem