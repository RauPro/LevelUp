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
