# api.py

from fastapi import FastAPI, APIRouter
from ..config import setup_config, get_ssl_certificates
from .websocket_handlers import WebsocketHandler
from .api_routes import APIRoutes
from topos.generations.openai_chat import OpenAIChatModel
from topos.generations.ollama_chat import OllamaChatModel
import os
import uvicorn

# Create the FastAPI application instance
app = FastAPI()

# Configure the application using settings from config.py
setup_config(app)

# Load API key for OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Make sure it's set in the environment variables.")

# Initialize models
openai_model = OpenAIChatModel(model_name="gpt-4", api_key=openai_api_key)  # OpenAI model with API key
ollama_model = OllamaChatModel(model_name="dolphin-llama3")  # Ollama model

# Select the model you want to use
selected_model = openai_model  # or ollama_model, depending on your requirements

# Set up WebSocket routes, passing the selected model to the handler
websocket_handler = WebsocketHandler(model=selected_model)
app.include_router(websocket_handler.router)

# Set up API routes, passing the selected model to the routes
api_routes = APIRoutes(model=selected_model)
app.include_router(api_routes.router)

def start_local_api():
    """Function to start the API in local mode."""
    print("\033[92mINFO:\033[0m     API docs available at: \033[1mhttp://0.0.0.0:13341/docs\033[0m")
    uvicorn.run(app, host="0.0.0.0", port=13341)

def start_web_api():
    """Function to start the API in web mode with SSL."""
    certs = get_ssl_certificates()
    uvicorn.run(app, host="0.0.0.0", port=13341, ssl_keyfile=certs['key_path'], ssl_certfile=certs['cert_path'])
