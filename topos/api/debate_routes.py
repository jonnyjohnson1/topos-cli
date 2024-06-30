# debate_routes.py

from typing import Dict

import os
from dotenv import load_dotenv

from fastapi import FastAPI, Form, Depends, APIRouter
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError

from datetime import datetime, timedelta, UTC
import json
import jwt
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Depends, status, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.security import OAuth2PasswordRequestForm
from typing import Union
from topos.FC.conversation_cache_manager import ConversationCacheManager
from topos.channel.debatesim import DebateSimulator
from topos.FC.conversation_cache_manager import ConversationCacheManager


load_dotenv()  # Load environment variables

router = APIRouter()
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"

conv_cache_manager = ConversationCacheManager()


def store_session(user_id, session_id):
    sessions = conv_cache_manager.load_from_cache(user_id, prefix="sessions") or {}
    sessions.setdefault(user_id, []).append(session_id)
    conv_cache_manager.save_to_cache(user_id, {user_id: sessions}, prefix="sessions")


def retrieve_sessions(user_id):
    sessions = conv_cache_manager.load_from_cache(user_id, prefix="sessions") or {}
    return sessions.get(user_id, [])


@router.post("/create_session")
async def create_session(token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id", "")
    except InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    session_id = f"session_{str(uuid4())}"
    # Store the session in your preferred storage
    store_session(user_id, session_id)
    return {"session_id": session_id}


@router.get("/sessions")
async def get_sessions(token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id", "")
    except InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    # Retrieve the sessions for the user
    sessions = retrieve_sessions(user_id)
    return {"sessions": sessions}


def save_accounts(account_dict, file_path='accounts.json'):
    with open(file_path, 'w') as file:
        json.dump(account_dict, file, indent=4)


def load_accounts(file_path='accounts.json') -> Dict[str, str]:
    try:
        with open(file_path, 'r') as file:
            account_dict = json.load(file)
    except FileNotFoundError:
        account_dict = {"userA": "pass",
                        "userB": "pass"}
    return account_dict


@router.post("/admin_set_accounts")
async def admin_set_all_accounts(request: Request):
    form_data = await request.form()
    accounts = {key: form_data[key] for key in form_data}
    save_accounts(accounts)
    return {"status": "success"}


# Add the route to issue JWT tokens
@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    accounts = load_accounts()

    # Validate user credentials
    if form_data.username not in accounts or accounts[form_data.username] != form_data.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    user_id = form_data.username  # Fetch the user_id based on your logic
    # Create JWT token
    token_data = {
        "user_id": user_id,
        "exp": datetime.now(UTC) + timedelta(hours=1)
    }
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}



# WebSocket endpoint with JWT validation
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: Union[str, None] = Query(default=None), session_id: Union[str, None] = Query(default=None)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
    except jwt.ExpiredSignatureError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    except jwt.InvalidTokenError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    if session_id is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    debate_simulator = await DebateSimulator.get_instance()
    # conv_cache_manager = ConversationCacheManager()

    print(f"Adding user {user_id} to session {session_id}")
    await debate_simulator.add_to_websocket_group(session_id, websocket)

    try:
        while True:
            print(f"Waiting for message from user {user_id}")
            data = await websocket.receive_json()

            message = data.get("message")
            generation_nonce = data.get("generation_nonce", str(uuid4()))
            current_topic = data.get("topic", "Unknown")
            processing_config = data.get("processing_config", {})
            user_name = data.get("user_name", "user")
            role = data.get("role", "user")

            print(f"Received message: {message}")

            # Set default debate values to calculate
            default_config = {
                "message_topic_analysis": True,
                "message_topic_mermaid_chart": True,
                "topic_cluster_num": True,
            }

            # Update default_config with provided processing_config, if any
            config = {**default_config, **processing_config}

            integrate_data = {
                "message": message,
                "user_id": user_id,
                "session_id": session_id,
                "generation_nonce": generation_nonce
            }

            # Integrate the message and start processing
            mermaid_ontology = await debate_simulator.integrate(token, json.dumps(integrate_data), debate_simulator.app_state, False)

            output_data = {'mermaid_ontology': json.dumps(mermaid_ontology)}

            if config['message_topic_analysis'] or config['message_topic_mermaid_chart']:
                print(f"\t[ save to conv cache :: conversation {session_id} ]")
                try:
                    output_data = {
                        'user_name': user_name,
                        'user_id': user_id,
                        'role': role,
                        'timestamp': datetime.now().isoformat(),
                        'message': message
                    }
                except Exception as e:
                    print("Error", e)

                if config.get('calculateModerationTags'):
                    message_topic_analysis = {}
                    topic_cluster_num = 0
                    message_topic_mermaid_chart = ""

                    output_data['analyses'] = {
                        'message_topic_analysis': message_topic_analysis,
                        'topic_cluster_num': topic_cluster_num,
                        'message_topic_mermaid_chart': message_topic_mermaid_chart
                    }
                conv_cache_manager.save_to_cache(session_id, output_data)
            else:
                print(f"\t[ save to conv cache :: conversation {session_id} ]")
                conv_cache_manager.save_to_cache(session_id, {
                    'user_name': user_name,
                    'user_id': user_id,
                    'role': role,
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                })

            # Send initial processing result back to the client
            await websocket.send_json({
                "status": "message_processed",
                "initial_analysis": output_data
            })

    except WebSocketDisconnect:
        await debate_simulator.remove_from_websocket_group(session_id, websocket)




# @router.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket, token: Union[str, None] = Query(default=None)):
#     # Validate JWT token
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#     except jwt.ExpiredSignatureError:
#         await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
#         return
#     except jwt.InvalidTokenError:
#         await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
#         return
#
#     await websocket.accept()
#
#     user_id = payload.get("sub")
#     session_id = f"session_{str(uuid4())}"
#
#     debate_simulator = await DebateSimulator.get_instance()
#
#     print(f"Adding user {user_id} to session {session_id}")
#     await debate_simulator.add_to_websocket_group(session_id, websocket)
#
#     try:
#         while True:
#             print(f"Waiting for message from user {user_id}")
#             data = await websocket.receive_text()
#             payload = json.loads(data)
#             message = payload.get("message")
#             message_id = str(uuid4())
#
#             print(f"Received message: {message}")
#             mermaid_diagram = await integrate_and_add_task(user_id, session_id, message_id, message, websocket)
#             await websocket.send_json({"status": "message_processed", "mermaid_diagram": mermaid_diagram,
#                                        "message_id": message_id})
#     except WebSocketDisconnect:
#         await debate_simulator.remove_from_websocket_group(session_id, websocket)
#
#
# async def integrate_and_add_task(user_id, session_id, message_id, message, websocket):
#     debate_simulator = await DebateSimulator.get_instance()
#
#     print(f"Integrating message: {message}")
#     # Build the ontology and get the mermaid diagram
#     mermaid_diagram = await debate_simulator.get_ontology(user_id, session_id, message_id, message)
#
#     # Prepare the message for broadcasting
#     broadcast_message = {
#         'user_id': user_id,
#         'session_id': session_id,
#         'message_id': message_id,
#         'mermaid_diagram': mermaid_diagram
#     }
#
#     # Add tasks to the queue
#     await debate_simulator.add_task(
#         {'type': 'ontology', 'user_id': user_id, 'session_id': session_id, 'message_id': message_id,
#          'message': message})
#     await debate_simulator.add_task({'type': 'broadcast', 'websocket': websocket, 'message': json.dumps(broadcast_message)})
#
#     return mermaid_diagram



# @router.post("/debate/per_message_base_analytics")
# async def process_message(request: Request):
#     """
#     Receives a payload with the current message, and returns per-message analyses:
#     1. topic similarity
#     2. topic category
#     3. mermaid chart related to message
#     4. the difference in overall topic similarity based on the message's contribution
#
#     returns:
#     return {
#         "status": "fetched_message_analysis",
#         "user_message": {
#             "message_id" :
#                 {
#                 'user_name': user_name,
#                 'user_id': user_id,
#                 'role': role,
#                 'message': message,
#                 'timestamp': datetime.now(),
#                 'analyses': {
#                     'message_topic_analysis': message_topic_analysis,
#                     'topic_cluster_num': topic_cluster_num,
#                     'message_topic_mermaid_chart': message_topic_mermaid_chart
#                     }
#                 }
#             }
#         }
#
#     """
#     try:
#         payload = await request.json()
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=400, detail="Invalid JSON")
#
#     conversation_id = payload.get("conversation_id")
#     message_id = payload.get("message_id")
#     message = payload.get("message")
#     current_topic = payload.get("topic", "Unknown")
#     processing_config = payload.get("processing_config", {})
#     user_id = payload.get("user_id", {})
#     user_name = payload.get("user_name",
#                             "user")  # let's just use the username for now to use to pull in the chatroom information
#     role = payload.get("role", "user")
#
#     # Set default debate values to calculate
#     default_config = {
#         "message_topic_analysis": True,
#         "message_topic_mermaid_chart": True,
#         "topic_cluster_num": True,
#     }
#
#     data = {"message_id": message_id, "message": message, "user_id": user_id, "session_id": session_id}
#
#     # Get the DebateSimulator singleton instance
#     debate_simulator = DebateSimulator()
#
#     # We're using the default debate_simulator app_state for now, but we can change this to a different AppState if we
#     # want to separate the debate logic from the general debate/chat logic
#     await debate_simulator.integrate(data, debate_simulator.app_state)
#
#     ## @note:@todo: so next steps are that the message_id is actually returned from debate.integrate which has kicked off an
#     # internal process to finally call debate.step like normal (with specific callbacks as intermediates)
#
#     # Update default_config with provided processing_config, if any
#     config = {**default_config, **processing_config}
#
#     conv_cache_manager = ConversationCacheManager()  # <- switch to other cache manager for debate specific information, or adapt this class to accept the debate schema
#     dummy_data = {}  # Replace with actual processing logic
#
#     if config['message_topic_analysis'] or config['message_topic_mermaid_chart']:
#         print(f"\t[ save to conv cache :: conversation {conversation_id}-{message_id} ]")
#         try:
#             dummy_data = {
#                 message_id:
#                     {
#                         'user_name': user_name,
#                         'user_id': user_id,
#                         'role': role,
#                         'timestamp': datetime.now(),
#                         'message': message
#                     }}
#         except Exception as e:
#             print("Error", e)
#
#         if config['calculateModerationTags']:
#             dummy_data[message_id]['analyses'] = {
#                 'message_topic_analysis': message_topic_analysis,
#                 'topic_cluster_num': topic_cluster_num,
#                 'message_topic_mermaid_chart': message_topic_mermaid_chart
#             }  # return analysis on message
#             conv_cache_manager.save_to_cache(conversation_id, dummy_data)
#             # Removing the keys from the nested dictionary
#         if message_id in dummy_data:
#             dummy_data[message_id].pop('message', None)
#             dummy_data[message_id].pop('timestamp', None)
#         # Sending first batch of user message analysis back to the UI
#         # return websocket.send_json({"status": "fetched_user_analysis", 'user_message': dummy_data})
#     else:
#         print(f"\t[ save to conv cache :: conversation {conversation_id}-{message_id} ]")
#         # Saving an empty dictionary for the messag id
#         conv_cache_manager.save_to_cache(conversation_id, {
#             message_id:
#                 {
#                     'user_name': user_name,
#                     'user_id': user_id,
#                     'role': role,
#                     'message': message,
#                     'timestamp': datetime.now(),
#                 }})
#         dummy_data = {
#             message_id:
#                 {
#                     'user_name': user_name,
#                     'user_id': user_id,
#                     'role': role,
#                     'message': message,
#                     'timestamp': datetime.now(),
#                 }}  # Replace with actual processing logic
#
#     return {"status": "fetched_message_analysis", "user_message": dummy_data}
#
#
# @router.post("/debate/all_messages_base_analytics")
# async def process_message(request: Request):
#     """
#     1. Receive payload with the conversation ID.
#     2. Load conversation history from ConversationCache
#
#     returns:
#     return {
#         "status": "fetched_message_analysis",
#         "user_message": {
#             "message_id" :
#                 {
#                 'user_name': user_name,
#                 'user_id': user_id,
#                 'role': role,
#                 'message': message,
#                 'timestamp': datetime.now(),
#                 'analyses': {
#                     'per_user_analyses': {
#                         "user_1": data,
#                         "user_2": data
#                     },
#                     'full_chat_analyses': data
#                     }
#                 }
#             }
#         }
#
#     """
#     try:
#         payload = await request.json()
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=400, detail="Invalid JSON")
#
#     conversation_id = payload.get("conversation_id")
#
#     # load conversation from conversation_id
#
#     # run analyses on conversation data
#     data = {}
#
#     # return the data
#
#     return {"status": "fetched_debate_analysis", "data": data}
