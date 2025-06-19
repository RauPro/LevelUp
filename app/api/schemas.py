import datetime
from uuid import UUID

from pydantic import BaseModel


class ChatStartResponse(BaseModel):
    session_id: UUID


class ChatMessageRequest(BaseModel):
    session_id: UUID
    message: str


class ChatMessage(BaseModel):
    sender: str
    message: str
    timestamp: datetime.datetime


class ChatMessageResponse(BaseModel):
    reply: str
    history: list[ChatMessage]


class ChatHistoryResponse(BaseModel):
    history: list[ChatMessage]


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
    user_prompt: str


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
