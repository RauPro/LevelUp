# LevelUp: AI-Powered Technical Interview Platform

LevelUp streamlines the technical interview process by using generative AI to create unique, topic-specific problems at varying difficulty levels. This addresses the challenge of candidates rehearsing common interview questions, providing a more effective and reliable evaluation of their true problem-solving abilities.

## 🚀 The Problem

The current technical recruitment process is highly predictable. Candidates often memorize solutions to common interview questions, which hinders a hiring manager's ability to assess their actual problem-solving skills. Additionally, creating unique and relevant questions for each candidate is a time-consuming and difficult task for interviewers to scale.

## ✨ Our Solution

LevelUp offers a web-based solution that generates complete and unique problem statements with a single click. The platform takes parameters like the desired topic (e.g., "Graph Theory") and difficulty level (e.g., "Medium") to produce a well-structured problem that includes a description, constraints, and sample inputs/outputs.

This is achieved using a Retrieval Augmented Generation (RAG) model that combines keyword and semantic search to retrieve relevant existing problems, which are then used by a Large Language Model (LLM) to generate a new, unique problem.

## Key Features for Hiring

-   **Improve Assessment Quality**: Evaluate candidates on their problem-solving skills, not their memorization abilities.
-   **Save Interviewer Time**: Instantly generate unique problems, significantly reducing preparation time for interviews.
-   **Ensure Fairness**: Provide standardized yet unique challenges for each candidate, ensuring a fair and consistent evaluation process.
-   **Customizable Content**: Tailor problems to specific topics and difficulty levels to match the requirements of the role.

## 🏛️ System Architecture

The LevelUp platform is built on a robust and scalable machine learning system architecture:

1.  **User Interface**: A simple and intuitive web-based interface where users can input their desired problem parameters.
2.  **Backend API Layer**: A FastAPI-based backend that handles user requests and orchestrates the problem generation workflow.
3.  **Data Ingestion**: A scraper ingests problems from sources like Codeforces. This data is then preprocessed and stored.
4.  **Metadata and Vector Storage**: Problem metadata is stored in a PostgreSQL database, while problem embeddings are stored in a Vector Database for efficient semantic retrieval.
5.  **RAG and LLM Pipeline**:
    * An **Embedding Generator** creates vector embeddings of the problems.
    * The **RAG Retriever** finds relevant problems from the vector database based on the user's query.
    * A **Prompt Generator** creates a prompt for the LLM based on the retrieved problems.
    * The **LLM** generates a unique problem based on the provided topic and style.

## 🛠️ Tech Stack

-   **Backend**: FastAPI
-   **ML Pipeline**: Retrieval Augmented Generation (RAG)
-   **Code Quality**: Ruff, Mypy
-   **Testing**: Pytest

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/LevelUp.git
    cd LevelUp
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Using UV for faster package management
    pip install uv
    uv venv
    # On Windows
    .venv\Scripts\activate
    # On Unix/MacOS
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    uv pip install -e .
    ```

## 🚀 Running the Application

1.  **Start the FastAPI server:**
    ```bash
    uvicorn app.main:app --reload
    ```

2.  **Access the API documentation:**
    -   Open your web browser and navigate to `http://127.0.0.1:8000/docs` for the Swagger UI documentation
    -   Or visit `http://127.0.0.1:8000/redoc` for ReDoc documentation

## 🧪 Running Tests

Run the tests using pytest:
```bash
pytest
```

## 🧹 Code Quality

Check code quality with Ruff:
```bash
ruff check .
```

Run type checking with Mypy:
```bash
mypy .
```

## 🤝 Contribution

We welcome contributions to the LevelUp project! If you have ideas for new features, improvements, or bug fixes, please feel free to open an issue or submit a pull request.
