import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatHistoryResponse, ChatMessage, ChatMessageRequest, ChatMessageResponse, ChatStartResponse

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


@router.post("/chat/message", response_model=ChatMessageResponse)
def post_message(request: ChatMessageRequest) -> ChatMessageResponse:
    """Receives a message, adds it to the history, and returns a reply."""
    if request.session_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Session not found")

    # Add user message to history
    user_message = ChatMessage(sender="user", message=request.message, timestamp=datetime.datetime.now())
    chat_histories[request.session_id].append(user_message)

    # Generate a dummy bot reply
    bot_reply = ChatMessage(sender="bot", message=f"Bot reply to: {request.message}", timestamp=datetime.datetime.now())
    chat_histories[request.session_id].append(bot_reply)

    return ChatMessageResponse(reply=bot_reply.message, history=chat_histories[request.session_id])


@router.get("/chat/history", response_model=ChatHistoryResponse)
def get_chat_history(session_id: UUID) -> ChatHistoryResponse:
    """Retrieves the chat history for a given session."""
    if session_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Session not found")

    return ChatHistoryResponse(history=chat_histories[session_id])
