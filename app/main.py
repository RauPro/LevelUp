from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ProblemRequest(BaseModel):
    """Request model for generating a problem."""

    user_prompt: str
    topic: str
    difficulty: str


class ProblemResponse(BaseModel):
    """Response model for a generated problem."""

    problem_id: int
    user_prompt: str
    title: str
    description: str
    topic: str
    difficulty: str


@app.get("/", response_model=dict[str, str])
def root() -> dict[str, str]:
    """Root endpoint for the API."""
    return {"message": "Welcome to LevelUp!"}


@app.post("/generate", response_model=ProblemResponse)
def generate_problem(request: ProblemRequest) -> ProblemResponse:
    """Generates a sample problem based on the user's request."""
    return ProblemResponse(problem_id=1, user_prompt=f"Here is requested prompt from user: {request.user_prompt}", title="Sample Problem", description="This is a sample problem generated based on your request.", topic=request.topic, difficulty=request.difficulty)
