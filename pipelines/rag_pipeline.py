# ml/rag_pipeline.py
import os
import json
from app.api.schemas import ProblemRequest, ProblemResponse
from ml.embedding_generator import problem_collection

# Import the official Mistral AI client
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
load_dotenv()

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
    The main RAG pipeline function using the Mistral AI client.
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

    api_key = os.getenv("MISTRAL_API_KEY")
    client = MistralClient(api_key=api_key)

    model_name = "mistral-tiny"

    messages = [
        ChatMessage(role="user", content=prompt)
    ]

    chat_response = client.chat(
        model=model_name,
        messages=messages,
    )
    import ast
    generated_content = chat_response.choices[0].message.content
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
                retry_messages.append(ChatMessage(role="assistant", content=generated_content))
                retry_messages.append(ChatMessage(role="user",
                                                  content="Your previous response couldn't be parsed. Please provide ONLY a valid Python dictionary with the problem data, without any additional text or formatting."))

                chat_response = client.chat(
                    model=model_name,
                    messages=retry_messages,
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