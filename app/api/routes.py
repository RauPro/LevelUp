from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    DifficultyLevel,
    ProblemRequest,
    ProblemTopic,
)
from data.sql_db.insert_grafana_data import save_evaluation_results
from data.sql_db.session_state_db import (
    get_session_state,
    get_session_states_by_criteria,
    save_session_state,
)
from ml.agent import app as agent_app
from ml.agent import create_initial_state
from ml.eval_mlflow import log_to_mlflow

router = APIRouter()


@router.get("/topics", response_model=list[str])
async def get_topics() -> list[str]:
    return [topic.value for topic in ProblemTopic]


@router.get("/difficulties", response_model=list[str])
async def get_difficulties() -> list[str]:
    return [level.value for level in DifficultyLevel]


@router.post("/problems/verified", response_model=dict)
async def generate_verified_problem(request: ProblemRequest) -> dict:
    """
    Generate a unique problem with verified test cases.

    This endpoint uses an AI agent that:
    1. Generates a programming problem
    2. Writes a solution for the problem
    3. Tests the solution against the test cases
    4. Only returns problems where the tests pass
    5. Logs metrics and state to MLflow for monitoring

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

        # Log results and artifacts to MLflow
        info_for_grafana = log_to_mlflow(final_state, state_history)
        # Save to database
        run_id, metrics, problem_attempts, code_attempts = info_for_grafana
        save_evaluation_results(
            run_id=run_id,
            metrics_dict=metrics,
            problem_attempts=problem_attempts,
            code_attempts=code_attempts,
        )

        # Save session state to database
        workflow_status = "completed" if final_state["tests_passed"] else "failed"
        save_session_state(thread_id=thread_id, state=final_state, run_id=run_id, workflow_status=workflow_status)

        # Check if tests passed
        if not final_state["tests_passed"]:
            raise ValueError("Could not generate a problem with passing test cases")

        # Convert agent's Problem format to ProblemResponse format
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

            # Create a problem response
            problem_response = {
                "id": thread_id,
                "title": f"Problem: {request.topic.value.title()}",
                "description": problem.description,
                "constraints": ["Time limit: 1 second", "Memory limit: 256 MB"],
                "examples": examples,
                "difficulty": request.difficulty,
                "topic": request.topic,
                "solution": final_state["code"] if final_state["code"] else "No solution available",
            }

            return {"response": problem_response}
        else:
            raise ValueError("No problem was generated")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate verified problem: {str(e)}") from e


@router.get("/sessions/{thread_id}", response_model=dict)
async def get_session_by_thread_id(thread_id: str) -> dict:
    """
    Retrieve a session state by thread ID.

    Args:
        thread_id: The unique thread identifier for the session

    Returns:
        dict: The session state data

    Raises:
        HTTPException: If the session is not found
    """
    session_data = get_session_state(thread_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session": session_data}


@router.get("/sessions", response_model=dict)
async def get_sessions(topic: str = None, difficulty: str = None, workflow_status: str = None, tests_passed: bool = None, limit: int = 50) -> dict:
    """
    Retrieve session states based on filtering criteria.

    Args:
        topic: Filter by problem topic (optional)
        difficulty: Filter by difficulty level (optional)
        workflow_status: Filter by workflow status (optional)
        tests_passed: Filter by test success status (optional)
        limit: Maximum number of sessions to return (default: 50, max: 100)

    Returns:
        dict: List of session states matching the criteria
    """
    # Validate limit
    if limit > 100:
        limit = 100

    sessions = get_session_states_by_criteria(topic=topic, difficulty=difficulty, workflow_status=workflow_status, tests_passed=tests_passed, limit=limit)

    return {"sessions": sessions, "count": len(sessions), "filters": {"topic": topic, "difficulty": difficulty, "workflow_status": workflow_status, "tests_passed": tests_passed, "limit": limit}}
