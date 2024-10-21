# models.py

from pydantic import BaseModel

class Message(BaseModel):
    content: str
    sender: str

class ModelConfig(BaseModel):
    model: str
    temperature: float
    
class MermaidChartPayload(BaseModel):
    message: str = None
    conversation_id: str
    full_conversation: bool = False
    model: str = "dolphin-llama3"
    provider: str = "ollama"
    api_key: str = "ollama"
    temperature: float = 0.04


class ConversationTopicsRequest(BaseModel):
    conversation_id: str
    model: str

class ConversationIDRequest(BaseModel):
    conversation_id: str
