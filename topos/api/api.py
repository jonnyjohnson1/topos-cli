from fastapi import FastAPI
from ..config import setup_config, get_ssl_certificates
import uvicorn

# Create the FastAPI application instance
app = FastAPI()

# Configure the application using settings from config.py
setup_config(app)

from .routers.server.system import router as system_router
from .routers.server.info import router as info_router
from .routers.analyze.graph import router as analyze_graph_router
from .routers.analyze.topics import router as analyze_topics_router
from .routers.analyze.summarize import router as analyze_summarize_router
from .routers.report.report import router as report_router
from .routers.image.image import router as image_router
from .routers.chat.chat import router as chat_router
from .routers.chat.p2p import router as p2p_router

# NEW ROUTER IMPORTS
app.include_router(system_router)
app.include_router(info_router)
app.include_router(analyze_graph_router)
app.include_router(analyze_topics_router)
app.include_router(analyze_summarize_router)
app.include_router(report_router)
app.include_router(image_router)
app.include_router(chat_router)
app.include_router(p2p_router)


"""

START API OPTIONS

There is the web option for networking with the online, webhosted version
There is the local option to connect the local apps to the Topos API (Grow debugging, and the Chat app)


"""


def start_local_api():
    """Function to start the API in local mode."""
    print("\033[92mINFO:\033[0m     API docs available at: \033[1mhttp://0.0.0.0:13341/docs\033[0m")
    uvicorn.run(app, host="0.0.0.0", port=13341)


def start_web_api():
    """Function to start the API in web mode with SSL."""
    certs = get_ssl_certificates()
    uvicorn.run(app, host="0.0.0.0", port=13341, ssl_keyfile=certs['key_path'], ssl_certfile=certs['cert_path'])
    

def start_hosted_service():
    """Function to start the API in web mode with SSL."""
    uvicorn.run(app, host="0.0.0.0", port=8000)