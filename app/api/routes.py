
from typing import List

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    DifficultyLevel,
    ProblemExample,
    ProblemRequest,
    ProblemResponse,
    ProblemTopic,
)

router = APIRouter()


@router.get("/topics", response_model=List[str])
async def get_topics() -> List[str]:

    return [topic.value for topic in ProblemTopic]


@router.get("/difficulties", response_model=List[str])
async def get_difficulties() -> List[str]:

    return [level.value for level in DifficultyLevel]


@router.post("/problems", response_model=ProblemResponse)
async def generate_problem(request: ProblemRequest) -> ProblemResponse:
    """
    Generate a unique problem based on the provided criteria.

    Args:
        request: ProblemRequest containing topic, difficulty, and optional keywords

    Returns:
        ProblemResponse: The generated problem

    Raises:
        HTTPException: If problem generation fails
    """
    # This is a mock implementation - in a real scenario, this would call
    # the RAG system to generate a unique problem
    try:
        # Mock response for demonstration purposes
        return ProblemResponse(
            id="prob123",
            title=f"{request.topic.value.title()} Challenge",
            description=f"This is a {request.difficulty.value} problem about"
                        f" {request.topic.value}.",
            constraints=["1 <= n <= 10^5", "Time Complexity: O(n)", "Space Complexity: O(1)"],
            examples=[
                ProblemExample(
                    input="[1, 2, 3, 4, 5]",
                    output="15",
                    explanation="Sum of all elements: 1 + 2 + 3 + 4 + 5 = 15"
                )
            ],
            difficulty=request.difficulty,
            topic=request.topic
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate problem: {str(e)}") from e


@router.get("/problems/{problem_id}", response_model=ProblemResponse)
async def get_problem(problem_id: str) -> ProblemResponse:
    """
    Retrieve a specific problem by ID.

    Args:
        problem_id: The unique identifier of the problem

    Returns:
        ProblemResponse: The requested problem

    Raises:
        HTTPException: If the problem is not found
    """
    # Mock implementation - in a real scenario, this would fetch from a database
    if problem_id != "prob123":
        raise HTTPException(status_code=404, detail="Problem not found")

    return ProblemResponse(
        id=problem_id,
        title="Arrays Challenge",
        description="This is a medium problem about arrays.",
        constraints=["1 <= n <= 10^5", "Time Complexity: O(n)", "Space Complexity: O(1)"],
        examples=[
            ProblemExample(
                input="[1, 2, 3, 4, 5]",
                output="15",
                explanation="Sum of all elements: 1 + 2 + 3 + 4 + 5 = 15"
            )
        ],
        difficulty=DifficultyLevel.MEDIUM,
        topic=ProblemTopic.ARRAYS
    )
