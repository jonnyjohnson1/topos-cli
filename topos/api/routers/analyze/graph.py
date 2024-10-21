import os
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from topos.FC.conversation_cache_manager import ConversationCacheManager

from ....services.generations_service.chat_gens import LLMController
from ....utilities.utils import create_conversation_string
from ....services.ontology_service.mermaid_chart import MermaidCreator
from ....models.models import MermaidChartPayload

import logging

router = APIRouter()

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


@router.post("/generate_mermaid_chart")
async def generate_mermaid_chart(payload: MermaidChartPayload):
    try:
        conversation_id = payload.conversation_id
        full_conversation = payload.full_conversation
        # model specifications
        model = payload.model
        provider = payload.provider# defaults to ollama right now
        api_key = payload.api_key
        temperature = payload.temperature

        llm_client = LLMController(model_name=model, provider=provider, api_key=api_key)

        mermaid_generator = MermaidCreator(llm_client)

        if full_conversation:
            cache_manager = cache_manager
            conv_data = cache_manager.load_from_cache(conversation_id)
            if conv_data is None:
                raise HTTPException(status_code=404, detail="Conversation not found in cache")
            print(f"\t[ generating mermaid chart :: {provider}/{model} :: full conversation ]")
            return {"status": "generating", "response": "generating mermaid chart", 'completed': False}
            # TODO: Complete this branch if needed

        else:
            message = payload.message
            if message:
                print(f"\t[ generating mermaid chart :: using model {model} ]")
                try:
                    mermaid_string = await mermaid_generator.get_mermaid_chart(message)
                    print(mermaid_string)
                    if mermaid_string == "Failed to generate mermaid":
                        return {"status": "error", "response": mermaid_string, 'completed': True}
                    else:
                        return {"status": "completed", "response": mermaid_string, 'completed': True}
                except Exception as e:
                    return {"status": "error", "response": f"Error: {e}", 'completed': True}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.websocket("/websocket_mermaid_chart")
async def meta_chat(websocket: WebSocket):
    """

    Generates a mermaid chart from a list of message.
    
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", None)
            conversation_id = payload["conversation_id"]
            full_conversation = payload.get("full_conversation", False)
            # model specifications
            model = payload.get("model", "dolphin-llama3")
            provider = payload.get('provider', 'ollama') # defaults to ollama right now
            api_key = payload.get('api_key', 'ollama')
            temperature = float(payload.get("temperature", 0.04))

            llm_client = LLMController(model_name=model, provider=provider, api_key=api_key)

            mermaid_generator = MermaidCreator(llm_client)
            # load conversation
            if full_conversation:
                cache_manager = cache_manager
                conv_data = cache_manager.load_from_cache(conversation_id)
                if conv_data is None:
                    raise HTTPException(status_code=404, detail="Conversation not found in cache")
                print(f"\t[ generating mermaid chart :: using model {model} :: full conversation ]")
                await websocket.send_json({"status": "generating", "response": "generating mermaid chart", 'completed': False})
                context = create_conversation_string(conv_data, 12)
                # TODO Complete this branch
            else:
                if message:
                    print(f"\t[ generating mermaid chart :: using model {model} ]")
                    await websocket.send_json({"status": "generating", "response": "generating mermaid chart", 'completed': False})
                    try:
                        mermaid_string = await mermaid_generator.get_mermaid_chart(message, websocket = websocket)
                        if mermaid_string == "Failed to generate mermaid":
                            await websocket.send_json({"status": "error", "response": mermaid_string, 'completed': True})
                        else:
                            await websocket.send_json({"status": "completed", "response": mermaid_string, 'completed': True})
                    except Exception as e:
                        await websocket.send_json({"status": "error", "response": f"Error: {e}", 'completed': True})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})
        await websocket.close()
    finally:
        await websocket.close()


