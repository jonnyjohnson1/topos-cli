from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import asynccontextmanager
import asyncio
from ..config import setup_config, get_ssl_certificates
from .websocket_handlers import router as websocket_router
from .api_routes import router as api_router
from .p2p_chat_routes import router as p2p_chat_router
from .debate_routes import router as debate_router
import uvicorn
from topos.services.messages.kafka_manager import KafkaManager


# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'chat_topic'

kafka_manager = KafkaManager(KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC)

# Create the FastAPI application instance
@asynccontextmanager
async def lifespan(app: FastAPI):
    await kafka_manager.start()
    consume_task = asyncio.create_task(kafka_manager.consume_messages(broadcast_message))
    yield
    consume_task.cancel()
    await kafka_manager.stop()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def broadcast_message(message, group_id):
    # Implement the logic to broadcast the message to all connected WebSocket clients in the group
    pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Process the received message and send it to Kafka
            await kafka_manager.send_message(group_id, data)
    except WebSocketDisconnect:
        # Handle disconnection
        pass

# Configure the application using settings from config.py
setup_config(app)

# Include routers from other parts of the application
app.include_router(api_router)
app.include_router(debate_router)
app.include_router(websocket_router)
app.include_router(p2p_chat_router)

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
