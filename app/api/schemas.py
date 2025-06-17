
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ProblemTopic(str, Enum):
    ARRAYS = "arrays"
    STRINGS = "strings"
    LINKED_LISTS = "linked_lists"
    TREES = "trees"
    GRAPHS = "graphs"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    SORTING = "sorting"
    SEARCHING = "searching"
    RECURSION = "recursion"
    BACKTRACKING = "backtracking"


class ProblemRequest(BaseModel):
    topic: ProblemTopic
    difficulty: DifficultyLevel
    keywords: Optional[List[str]] = Field(default=None,
                                          description="Optional keywords to tailor the problem")


class ProblemExample(BaseModel):
    input: str
    output: str
    explanation: Optional[str] = None


class ProblemResponse(BaseModel):
    id: str
    title: str
    description: str
    constraints: List[str]
    examples: List[ProblemExample]
    difficulty: DifficultyLevel
    topic: ProblemTopic
