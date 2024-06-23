from datetime import datetime
import json
from fastapi import APIRouter, HTTPException
from requests import Request
import tkinter as tk
from tkinter import filedialog
from topos.FC.conversation_cache_manager import ConversationCacheManager
from topos.channel.debatesim import DebateSimulator

router = APIRouter()


def integrate_and_add_task(user_id, session_id, message_id, message, websocket):
    debate_simulator = DebateSimulator.get_instance()

    # Build the ontology and get the mermaid diagram
    mermaid_diagram = debate_simulator.build_ontology(user_id, session_id, message_id, message)

    # Prepare the message for broadcasting
    broadcast_message = {
        'user_id': user_id,
        'session_id': session_id,
        'message_id': message_id,
        'mermaid_diagram': mermaid_diagram
    }

    # Add tasks to the queue
    debate_simulator.add_task(
        {'type': 'ontology', 'user_id': user_id, 'session_id': session_id, 'message_id': message_id,
         'message': message})
    debate_simulator.add_task({'type': 'broadcast', 'websocket': websocket, 'message': json.dumps(broadcast_message)})

    return mermaid_diagram

@router.post("/debate/per_message_base_analytics")
async def process_message(request: Request):
    """

    Receives a payload with the current message, and returns per-message analyses:
    1. topic similarity
    2. topic category
    3. mermaid chart related to message
    4. the difference in overall topic similarity based on the message's contribution

    returns:
    return {
        "status": "fetched_message_analysis",
        "user_message": {
            "message_id" :
                {
                'user_name': user_name,
                'user_id': user_id,
                'role': role,
                'message': message,
                'timestamp': datetime.now(),
                'analyses': {
                    'message_topic_analysis': message_topic_analysis,
                    'topic_cluster_num': topic_cluster_num,
                    'message_topic_mermaid_chart': message_topic_mermaid_chart
                    }
                }
            }
        }

    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    pay

    conversation_id = payload.get("conversation_id")
    message_id = payload.get("message_id")
    message = payload.get("message")
    current_topic = payload.get("topic", "Unknown")
    processing_config = payload.get("processing_config", {})
    user_id = payload.get("user_id", {})
    user_name = payload.get("user_name",
                            "user")  # let's just use the username for now to use to pull in the chatroom information
    role = payload.get("role", "user")

    # Set default debate values to calculate
    default_config = {
        "message_topic_analysis": True,
        "message_topic_mermaid_chart": True,
        "topic_cluster_num": True,
    }

    data = {"message_id": message_id, "message": message, "user_id": user_id, "session_id": session_id}

    # Get the DebateSimulator singleton instance
    debate_simulator = DebateSimulator()

    # We're using the default debate_simulator app_state for now, but we can change this to a different AppState if we
    # want to separate the debate logic from the general debate/chat logic
    await debate_simulator.integrate(data, debate_simulator.app_state)

    ## @note:@todo: so next steps are that the message_id is actually returned from debate.integrate which has kicked off an
    # internal process to finally call debate.step like normal (with specific callbacks as intermediates)

    # Update default_config with provided processing_config, if any
    config = {**default_config, **processing_config}

    conv_cache_manager = ConversationCacheManager()  # <- switch to other cache manager for debate specific information, or adapt this class to accept the debate schema
    dummy_data = {}  # Replace with actual processing logic

    if config['message_topic_analysis'] or config['message_topic_mermaid_chart']:
        print(f"\t[ save to conv cache :: conversation {conversation_id}-{message_id} ]")
        try:
            dummy_data = {
                message_id:
                    {
                        'user_name': user_name,
                        'user_id': user_id,
                        'role': role,
                        'timestamp': datetime.now(),
                        'message': message
                    }}
        except Exception as e:
            print("Error", e)

        if config['calculateModerationTags']:
            dummy_data[message_id]['analyses'] = {
                'message_topic_analysis': message_topic_analysis,
                'topic_cluster_num': topic_cluster_num,
                'message_topic_mermaid_chart': message_topic_mermaid_chart
            }  # return analysis on message
            conv_cache_manager.save_to_cache(conversation_id, dummy_data)
            # Removing the keys from the nested dictionary
        if message_id in dummy_data:
            dummy_data[message_id].pop('message', None)
            dummy_data[message_id].pop('timestamp', None)
        # Sending first batch of user message analysis back to the UI
        # return websocket.send_json({"status": "fetched_user_analysis", 'user_message': dummy_data})
    else:
        print(f"\t[ save to conv cache :: conversation {conversation_id}-{message_id} ]")
        # Saving an empty dictionary for the messag id
        conv_cache_manager.save_to_cache(conversation_id, {
            message_id:
                {
                    'user_name': user_name,
                    'user_id': user_id,
                    'role': role,
                    'message': message,
                    'timestamp': datetime.now(),
                }})
        dummy_data = {
            message_id:
                {
                    'user_name': user_name,
                    'user_id': user_id,
                    'role': role,
                    'message': message,
                    'timestamp': datetime.now(),
                }}  # Replace with actual processing logic

    return {"status": "fetched_message_analysis", "user_message": dummy_data}


@router.post("/debate/all_messages_base_analytics")
async def process_message(request: Request):
    """

    1. Receive payload with the conversation ID.
    2. Load conversation history from ConversationCache

    returns:
    return {
        "status": "fetched_message_analysis",
        "user_message": {
            "message_id" :
                {
                'user_name': user_name,
                'user_id': user_id,
                'role': role,
                'message': message,
                'timestamp': datetime.now(),
                'analyses': {
                    'per_user_analyses': {
                        "user_1": data,
                        "user_2": data
                    },
                    'full_chat_analyses': data
                    }
                }
            }
        }

    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    conversation_id = payload.get("conversation_id")

    # load conversation from conversation_id

    # run analyses on conversation data
    data = {}

    # return the data

    return {"status": "fetched_debate_analysis", "data": data}
