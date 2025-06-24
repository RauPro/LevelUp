import os
from typing import Dict, List, Any

# Import the collection from embedding_generator
from ml.embedding_generator import problem_collection


class RagRetriever:
    """
    A RAG (Retrieval-Augmented Generation) retriever that uses a vector database
    to find similar programming problems to use as context for generating new problems.
    """

    def __init__(self, n_results: int = 3):
        """
        Initialize the RAG retriever.

        Args:
            n_results: Number of similar problems to retrieve
        """
        self.n_results = n_results

    def retrieve(self, topic: str, difficulty: str, user_prompt: str) -> List[Dict[str, Any]]:
        """
        Retrieve similar problems based on the query.

        Args:
            topic: Problem topic
            difficulty: Problem difficulty level
            user_prompt: Additional user context

        Returns:
            List of retrieved problems with their documents and metadata
        """
        query_text = f"{topic} {difficulty} {user_prompt}"

        retrieved = problem_collection.query(
            query_texts=[query_text],
            n_results=self.n_results,
        )

        retrieved_problems = [
            {"documents": doc, "metadatas": meta}
            for doc, meta in zip(retrieved["documents"][0], retrieved["metadatas"][0])
        ]

        return retrieved_problems

    def generate_prompt(self, topic: str, difficulty: str, retrieved_problems: list) -> str:
        """Creates a detailed prompt for the LLM using RAG approach."""
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
            "Now, generate a brand new problem. The problem should be in a JSON format with the following structure:\n"
            "{\n"
            "  'description': '...',  // Detailed problem description\n"
            "  'tests': [\n"
            "    {\n"
            "      'input': '...',\n"
            "      'output': '...'\n"
            "    },\n"
            "    // You can include multiple examples\n"
            "  ]\n"
            "}\n"
            "Make sure your problem is unique and appropriate for the difficulty level. Include at least two test cases with input and output examples.\n"
        )

        return prompt

# Create a default instance for easy importing
default_retriever = RagRetriever()
