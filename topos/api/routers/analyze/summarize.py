from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
import json
import os

from ....services.generations_service.chat_gens import LLMController
from ....utils.utils import create_conversation_string

# cache database
from topos.services.database.conversation_cache_manager import ConversationCacheManager

import logging

db_config = {
            "dbname": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
            "host": os.getenv("POSTGRES_HOST"),
            "port": os.getenv("POSTGRES_PORT")
        }

logging.info(f"Database configuration: {db_config}")

use_postgres = True
if use_postgres:
    cache_manager = ConversationCacheManager(use_postgres=True, db_config=db_config)
else:
    cache_manager = ConversationCacheManager()

router = APIRouter()

@router.websocket("/websocket_chat_summary")
async def meta_chat(websocket: WebSocket):
    """

    Generates a summary of the conversation oriented around a given focal point.
    
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            conversation_id = payload["conversation_id"]
            subject = payload.get("subject", "knowledge")
            temperature = float(payload.get("temperature", 0.04))
            
            # model specifications
            model = payload.get("model", "solar")
            provider = payload.get('provider', 'ollama') # defaults to ollama right now
            api_key = payload.get('api_key', 'ollama')

            llm_client = LLMController(model_name=model, provider=provider, api_key=api_key)


            # load conversation
            cache_manager = cache_manager
            conv_data = cache_manager.load_from_cache(conversation_id)
            if conv_data is None:
                raise HTTPException(status_code=404, detail="Conversation not found in cache")

            context = create_conversation_string(conv_data, 12)

            print(f"\t[ generating summary :: model {model} :: subject {subject}]")

            # Set system prompt
            system_prompt = "PRESENT CONVERSATION:\n-------<context>" + context + "\n-------\n"
            query = f"""Summarize this conversation. Frame your response around the subject of {subject}"""
            
            msg_history = [{'role': 'system', 'content': system_prompt}]

            # Append the present message to the message history
            simplified_message = {'role': "user", 'content': query}
            msg_history.append(simplified_message)

            # Processing the chat
            output_combined = ""
            for chunk in llm_client.stream_chat(msg_history, temperature=temperature):
                try:
                    output_combined += chunk
                    await websocket.send_json({"status": "generating", "response": output_combined, 'completed': False})
                except Exception as e:
                    print(e)
                    await websocket.send_json({"status": "error", "message": str(e)})
                    await websocket.close()
            # Send the final completed message
            await websocket.send_json(
                {"status": "completed", "response": output_combined, "completed": True})

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})
        await websocket.close()
