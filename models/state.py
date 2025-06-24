from typing import TypedDict, Optional

from pydantic import BaseModel

from app.api.schemas import ProblemTopic, DifficultyLevel


class Problem(BaseModel):
    description: str
    tests: list[dict[str, str]]


class SessionState(TypedDict):
    user_prompt: str
    topic: ProblemTopic
    difficulty: DifficultyLevel
    problem: Optional[Problem]
    code: Optional[str]
    tests_passed: bool
    code_attempts: int
    problem_attempts: int