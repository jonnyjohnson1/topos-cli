from fastapi import FastAPI
from ..config import setup_config, get_ssl_certificates
import uvicorn

from .server import app as chat_app

def start_messenger_server():
    """Function to start the API in local mode."""
    print("\033[92mINFO:\033[0m     API docs available at: \033[1mhttp://127.0.0.1:13394/docs\033[0m")
    uvicorn.run(chat_app, host="127.0.0.1", port=13394)

# start through zrok
# uvicorn main:app --host 127.0.0.1 --port 13394 & zrok expose http://localhost:13394
