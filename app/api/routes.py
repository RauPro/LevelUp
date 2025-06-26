import asyncio
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    ChatHistoryResponse,
    ChatMessage,
    ChatStartResponse,
    DifficultyLevel,
    ProblemExample,
    ProblemRequest,
    ProblemResponse,
    ProblemTopic,
)
from data.sql_db.insert_grafana_data import save_evaluation_results
from ml.agent import app as agent_app
from ml.agent import create_initial_state
from ml.eval_mlflow import log_to_mlflow
from pipelines.rag_pipeline import generate_new_problem

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

    # user_message = ChatMessage(sender="user", message=request.message, timestamp=datetime.datetime.now())
    # chat_histories[request.session_id].append(user_message)

    bot_reply = await generate_problem(request)
    # chat_histories[request.session_id].append(bot_reply)

    return bot_reply


@router.get("/chat/history", response_model=ChatHistoryResponse)
def get_chat_history(session_id: UUID) -> ChatHistoryResponse:
    """Retrieves the chat history for a given session."""
    if session_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Session not found")

    return ChatHistoryResponse(history=chat_histories[session_id])


@router.get("/topics", response_model=list[str])
async def get_topics() -> list[str]:
    return [topic.value for topic in ProblemTopic]


@router.get("/difficulties", response_model=list[str])
async def get_difficulties() -> list[str]:
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
    try:
        return {"response": await generate_new_problem(request)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate problem: {str(e)}") from e


@router.post("/problems/verified", response_model=dict)
async def generate_verified_problem(request: ProblemRequest) -> dict:
    """
    Generate a unique problem with verified test cases.

    This endpoint uses an AI agent that:
    1. Generates a programming problem
    2. Writes a solution for the problem
    3. Tests the solution against the test cases
    4. Only returns problems where the tests pass
    5. Logs metrics and state to MLflow for monitoring (asynchronously)

    Args:
        request: ProblemRequest containing topic, difficulty, and user_prompt

    Returns:
        dict: A response containing the verified problem and MLflow tracking information

    Raises:
        HTTPException: If problem generation fails or no valid problem could be generated
    """
    try:
        # Create a unique ID for this workflow run to track state history
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Create initial state for the agent
        initial_state = create_initial_state(
            user_prompt=request.user_prompt,
            topic=request.topic,
            difficulty=request.difficulty,
        )

        # Invoke the agent to get the final state
        final_state = agent_app.invoke(initial_state, config=config)

        # Retrieve the full state history using the thread_id
        state_history = agent_app.get_state_history(config)

        # Check if tests passed early
        if not final_state["tests_passed"]:
            raise ValueError("Could not generate a problem with passing test cases")

        # Prepare the response first
        if final_state["problem"]:
            problem = final_state["problem"]

            # Format examples from tests
            examples = []
            for test in problem.tests:
                examples.append(
                    {
                        "input": test["input"],
                        "output": test["output"],
                        "explanation": "Test case for the problem.",
                    }
                )

            # Generate a unique problem ID and run ID for tracking
            problem_id = str(uuid4())
            run_id = f"pending_{uuid4()}"  # Temporary run ID until MLflow completes

            # Create a problem response
            problem_response = {
                "id": problem_id,
                "title": f"Problem: {request.topic.value.title()}",
                "description": problem.description,
                "constraints": ["Time limit: 1 second", "Memory limit: 256 MB"],
                "examples": examples,
                "difficulty": request.difficulty,
                "topic": request.topic,
                "solution": final_state["code"] if final_state["code"] else "No solution available",
                "mlflow_run_id": run_id,  # Will be updated asynchronously
            }

            # Start MLflow logging in the background (fire and forget)
            asyncio.create_task(log_and_save_async(final_state, state_history))

            return {"response": problem_response}
        else:
            raise ValueError("No problem was generated")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate verified problem: {str(e)}") from e


async def log_and_save_async(final_state, state_history):
    """
    Asynchronously log results to MLflow and save to database.
    This runs in the background without blocking the API response.
    """
    try:
        # Run the heavy MLflow logging in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        info_for_grafana = await loop.run_in_executor(
            None, log_to_mlflow, final_state, state_history
        )

        # Save to database - unpack the tuple properly
        if len(info_for_grafana) == 4:
            run_id, metrics, problem_attempts, code_attempts = info_for_grafana
            await loop.run_in_executor(
                None,
                save_evaluation_results,
                run_id,
                metrics,
                problem_attempts,
                code_attempts,
            )
            print(f"✅ Successfully logged to MLflow and database. Run ID: {run_id}")
        else:
            print(f"⚠️ Unexpected return format from log_to_mlflow: {info_for_grafana}")

    except Exception as e:
        # Log the error but don't fail the API response
        print(f"❌ Error in background logging: {str(e)}")


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
        examples=[ProblemExample(input="[1, 2, 3, 4, 5]", output="15", explanation="Sum of all elements: 1 + 2 + 3 + 4 + 5 = 15")],
        difficulty=DifficultyLevel.MEDIUM,
        topic=ProblemTopic.ARRAYS,
    )
