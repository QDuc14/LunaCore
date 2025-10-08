from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str