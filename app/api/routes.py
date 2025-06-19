import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatHistoryResponse, ChatMessage, ChatMessageRequest, ChatMessageResponse, ChatStartResponse
from pipelines.rag_pipeline import generate_new_problem
from app.api.schemas import (
    DifficultyLevel,
    ProblemExample,
    ProblemRequest,
    ProblemResponse,
    ProblemTopic,
)
router = APIRouter()

# In-memory storage for chat histories (for prototyping)
###### For future it should be on postgres ########
chat_histories: dict[UUID, list[ChatMessage]] = {}


@router.post("/chat/start", response_model=ChatStartResponse)
def start_chat_session() -> ChatStartResponse:
    """Starts a new chat session and returns a session ID."""
    session_id = uuid4()
    chat_histories[session_id] = []
    print(f"New chat session started with ID: {session_id}")
    return ChatStartResponse(session_id=session_id)


@router.post("/chat/message", response_model=dict)
async def post_message(request: ProblemRequest) -> dict:
    """Receives a message, adds it to the history, and returns a reply."""
    # if request.session_id not in chat_histories:
    #     raise HTTPException(status_code=404, detail="Session not found")

    # Add user message to history
    # user_message = ChatMessage(sender="user", message=request.message, timestamp=datetime.datetime.now())
    # chat_histories[request.session_id].append(user_message)

    # Generate a dummy bot reply
    bot_reply = await generate_problem(request)
    #chat_histories[request.session_id].append(bot_reply)

    return bot_reply


@router.get("/chat/history", response_model=ChatHistoryResponse)
def get_chat_history(session_id: UUID) -> ChatHistoryResponse:
    """Retrieves the chat history for a given session."""
    if session_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Session not found")

    return ChatHistoryResponse(history=chat_histories[session_id])



from typing import List

from fastapi import APIRouter, HTTPException





@router.get("/topics", response_model=List[str])
async def get_topics() -> List[str]:

    return [topic.value for topic in ProblemTopic]


@router.get("/difficulties", response_model=List[str])
async def get_difficulties() -> List[str]:

    return [level.value for level in DifficultyLevel]


@router.post("/problems", response_model=dict)
async def generate_problem(request: ProblemRequest) -> dict:
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
        return {"response": await generate_new_problem(request)}
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
