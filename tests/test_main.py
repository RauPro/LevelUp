

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client() -> TestClient:
    """
    Create a test client for the FastAPI application.

    Returns:
        TestClient: A test client for the FastAPI application
    """
    return TestClient(app)


def test_root_endpoint(client: TestClient) -> None:
    """
    Test the root endpoint of the API.

    Args:
        client: The test client for the FastAPI application
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Welcome to the LevelUp API" in data["message"]


def test_get_topics(client: TestClient) -> None:
    """
    Test the endpoint that returns available topics.

    Args:
        client: The test client for the FastAPI application
    """
    response = client.get("/api/topics")
    assert response.status_code == 200
    topics = response.json()
    assert isinstance(topics, list)
    assert "arrays" in topics
    assert "graphs" in topics


def test_get_difficulties(client: TestClient) -> None:
    """
    Test the endpoint that returns available difficulty levels.

    Args:
        client: The test client for the FastAPI application
    """
    response = client.get("/api/difficulties")
    assert response.status_code == 200
    difficulties = response.json()
    assert isinstance(difficulties, list)
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "hard" in difficulties


def test_generate_problem(client: TestClient) -> None:
    """
    Test the problem generation endpoint.

    Args:
        client: The test client for the FastAPI application
    """
    payload = {
        "topic": "arrays",
        "difficulty": "medium",
        "keywords": ["sorting", "searching"]
    }
    response = client.post("/api/problems", json=payload)
    assert response.status_code == 200
    problem = response.json()
    assert problem["id"] == "prob123"
    assert problem["topic"] == "arrays"
    assert problem["difficulty"] == "medium"
    assert len(problem["constraints"]) > 0
    assert len(problem["examples"]) > 0


def test_get_problem(client: TestClient) -> None:
    """
    Test retrieving a specific problem by ID.

    Args:
        client: The test client for the FastAPI application
    """
    response = client.get("/api/problems/prob123")
    assert response.status_code == 200
    problem = response.json()
    assert problem["id"] == "prob123"
    assert problem["topic"] == "arrays"
    assert problem["difficulty"] == "medium"


def test_get_nonexistent_problem(client: TestClient) -> None:
    """
    Test retrieving a nonexistent problem.

    Args:
        client: The test client for the FastAPI application
    """
    response = client.get("/api/problems/nonexistent")
    assert response.status_code == 404
    assert "Problem not found" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])