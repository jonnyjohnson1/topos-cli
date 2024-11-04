import asyncio
import datetime
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from fastapi.concurrency import asynccontextmanager
from topos.services.messages.group_management_service import GroupManagementService
from topos.services.messages.missed_message_service import MissedMessageService
from topos.utilities.utils import generate_deci_code, generate_group_name
from pydantic import BaseModel
# MissedMessageRequest model // subject to change
class MissedMessagesRequest(BaseModel):
    user_id: str
    # last_sync_time: datetime

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'chat_topic'


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
       self.active_connections: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        
    def register(self, websocket: WebSocket,user_id:str):
        self.active_connections[websocket] = user_id

    async def disconnect(self, websocket: WebSocket):
        user_id = self.active_connections[websocket]
        user = group_management_service.get_user_by_id(user_id)
        username = user['username']
        group_management_service.set_user_last_seen_online(user_id)
        users_list_of_groups =  group_management_service.get_user_groups(user_id)
        for group in users_list_of_groups:
        
            disconnect_message = f"{username} left the chat"
            message = { 
                        "message_id": generate_deci_code(16),
                        "message_type": "server",
                        "username": username,
                        "from_user_id":user_id,
                        "session_id": group["group_id"],
                        "message": disconnect_message,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
            await producer.send_and_wait(KAFKA_TOPIC, key=group["group_id"].encode('utf-8'),value=json.dumps(message).encode('utf-8')) 
        print(f"removing {user_id}")
        try:    
            del self.active_connections[websocket]
            print("Successfully Disconnected")
        except:
            print("Disconnect Failed. Error encountered. Check Logs")

    async def broadcast(self, from_user_id:str,message, group_id:str):#
        print(message)
        if(group_management_service.get_group_by_id(group_id=group_id)):
            print(f"{group_id} exists" )
            group_users_info = group_management_service.get_group_users(group_id=group_id)
            group_user_ids = [user['user_id'] for user in group_users_info]
            print(group_users_info)
            for connection,user_id in self.active_connections.items():
                print(f"Testing {user_id}")
                if(user_id in group_user_ids):
                    print(f"{user_id} is in {group_id}")
                    print(f"sending to {user_id}")
                    print(f"connection state is {connection.application_state}")
                    print(f"Sending message: {message}")
                    await connection.send_json(message)
                print("next connection")

db_config = {
            "dbname": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
            "host": os.getenv("POSTGRES_HOST"),
            "port": os.getenv("POSTGRES_PORT")
        }

group_management_service = GroupManagementService(db_params=db_config)
manager = ConnectionManager()

producer = None
consumer = None


async def consume_messages():
    async for msg in consumer:
        # print(msg.offset)
        message = json.loads(msg.value.decode('utf-8'))
        group_id = msg.key.decode('utf-8')
        await manager.broadcast(message=message,from_user_id=message["from_user_id"],group_id=group_id)
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    # Kafka producer
    global producer
    global consumer
    
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)

    # Kafka consumer
    consumer = AIOKafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    # group_id="chat_group"
    )

    await producer.start()
    await consumer.start()
    # https://stackoverflow.com/questions/46890646/asyncio-weirdness-of-task-exception-was-never-retrieved
    # we need to keep a reference of this task alive else it will stop the consume task, there has to be a live refference for this to work
    consume_task = asyncio.create_task(consume_messages())
    yield
    # Clean up the ML models and release the resources
    consume_task.cancel()
    await producer.stop()
    await consumer.stop()

# FastAPI app
app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print("client joined") # think about what needs to be done when the client joins like missed message services etc
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                payload = json.loads(data)
                print(payload)
                # get user details and associate with ws? we a separate client message to declare its identity after a potential disconnect, 1 time after 
                manager.register(websocket,payload['user_id'])
                if(group_management_service.get_user_by_id(payload["user_id"])== None):
                    group_management_service.create_user(username=payload["username"],user_id=payload["user_id"])
                message_type = payload['message_type']
                print(message_type)
                group_management_service.set_user_last_seen_online(payload['user_id'])
                if message_type == "create_server":
                    group_name = generate_group_name() # you can create group name on the frontend , this is just a basic util that can be swapped out if needed
                    group_id = group_management_service.create_group(group_name=group_name)
                    group_management_service.add_user_to_group(payload["user_id"],group_id=group_id)
                    print(f"[ client created chat :: group : {"group_name " + group_name + " : gid:"+ group_id} ]")
                    prompt_message = f"{payload["username"]} created the chat"
                    message = {
                        "message_type": "server",
                        "from_user_id": payload["user_id"],
                        "session_id": group_id,
                        "message": prompt_message,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    await producer.send_and_wait(KAFKA_TOPIC, key=group_id.encode('utf-8'),value=json.dumps(message).encode('utf-8'))
                elif message_type == "join_server":
                    group_id = payload['session_id']
                    user_id = payload['user_id']
                    username = payload['username']
                    # see if session exists 
                    if(group_management_service.get_group_by_id(group_id=group_id) == None):
                        await websocket.send_json({"error": "Invalid session"})
                    else: 
                        group_management_service.add_user_to_group(user_id=user_id,group_id=group_id)
                        join_message = f"{username} joined the chat"
                        print("Hells bells")
                        print(join_message)
                        message = {
                                "message_type": "server",
                                "from_user_id": payload["user_id"],
                                "session_id": group_id,
                                "message": join_message,
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                        await producer.send_and_wait(KAFKA_TOPIC, key=group_id.encode('utf-8'),value= json.dumps(message).encode('utf-8'))
                else:
                    print("RECEIVED: ", payload)
                    group_id = payload['session_id']
                    user_id = payload['user_id']
                    message_id = payload['message_id'] # generate_deci_code(16)
                    user = group_management_service.get_user_by_id(user_id)
                    message = {
                                "message_type": "user",
                                "message_id": message_id,
                                "from_user_id": user_id,
                                "username": user['username'],
                                "session_id": group_id,
                                "message": payload["content"]["text"],
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                    if (group_management_service.get_group_by_id(group_id=group_id)):
                        print(f"sending {group_id}")
                        await producer.send_and_wait(KAFKA_TOPIC, key=group_id.encode('utf-8'),value=json.dumps(message).encode('utf-8'))
                    else:
                        await websocket.send_json({"error": "Invalid session"})
       
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except asyncio.TimeoutError:
        print("client disconnected due to timeout")
        await manager.disconnect(websocket)
    except Exception as e:
        print(f"client disconnected due to error: {e}")
        await manager.disconnect(websocket)


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI aiokafka WebSocket Chat Server"}

@app.post("/chat/missed-messages")
async def get_missed_messages(request: MissedMessagesRequest):
    # get the user id and the pass it to the missed message service and then invoke it 
    missed_message_service = MissedMessageService()
    return await missed_message_service.get_missed_messages(user_id=request.user_id,group_management_service=group_management_service)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=13394)


""" Message JSON Schema

{ 
"message_id": "<unique_id>", 
"message_type": "<type>", // OPTIONS: user, ai, server 
“num_participants”: <int>,
"content": 
	{ 
	  "sender_id": "<id>",
	  "conversation_id": "<id>", 
	  "username": "<name>", 
	  "text": "<message_content>" 
	}, 
"timestamp": "<datetime>", 
"metadata": { 
	"priority": "<level>", // e.g., normal, high 
	"tags": ["<tag1>", "<tag2>"] 
	}, 
"attachments": [ 
	{ 
	  "file_name": "<file_name>", 
	  "file_type": "<type>", 
	  "url": "<file_url>" 
	} 
] }

"""
"""
create server message 
{
    "message_type": "create_server", 
    "num_participants": "5", 
    "host_name": "anshul",
    "user_id": "1",
    "created_at": "t0",
    "username": "anshul"
}
"""

"""
Revising the message format 
{ 
    "message_id": "69", 
    "message_type": "user", 
    "user_id": "2",
    "username": "jonny", 
    "session_id":"961198",
    "content": 
    { 
        "metadata": { 
            "priority": "<level>",  
            "tags": ["<tag1>", "<tag2>"] 
        }, 
        "attachments": [ 
                { 
                    "file_name": "<file_name>", 
                    "file_type": "<type>", 
                    "url": "<file_url>" 
                } 
            ],
        "text": "kafka chatserver works" 
    }, 
    "timestamp": "t5" 
}
"""

"""
Notes:
WE do not need to pass on any information like username instead it should probably be display name associated with a specific group 
Right now it is being treated as a solid username and not display name 
"""