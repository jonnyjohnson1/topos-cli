# models.py

from pydantic import BaseModel


class Message(BaseModel):
    content: str
    sender: str


class ModelConfig(BaseModel):
    model: str
    temperature: float