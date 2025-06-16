from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to LevelUp!"}


def test_generate_problem() -> None:
    payload = {"user_prompt": "Test prompt", "topic": "Graph Theory", "difficulty": "Medium"}
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["topic"] == "Graph Theory"
    assert data["difficulty"] == "Medium"
    assert "Sample" in data["title"]
