from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..generations.ollama_chat import stream_chat
from topos.FC.semantic_compression import SemanticCompression
from ..config import get_openai_api_key
import json

router = APIRouter()


@router.websocket("/websocket_chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload["message"]
            message_history = payload["message_history"]
            model = payload.get("model", "solar")
            temperature = float(payload.get("temperature", 0.04))
            current_topic = payload.get("topic", "Unknown")

            # Set system prompt
            has_topic = False
            
            if current_topic != "Unknown":
                has_topic = True
                prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is {current_topic}.\n"

            system_prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is unknown, so try not to make any judgements thus far - only re-express the input words in your own style:"
            user_prompt = ""
            if message_history:
                # Add the message history prior to the message
                user_prompt += '\n'.join(msg['role'] + ": " + msg['content'] for msg in message_history)

            print(f"\t[ system prompt :: {system_prompt} ]")
            print(f"\t[ user prompt :: {user_prompt} ]")
            simp_msg_history = [{'role': 'system', 'content': system_prompt}]

            # Simplify message history to required format
            for message in message_history:
                simplified_message = {'role': message['role'], 'content': message['content']}
                if 'images' in message:
                    simplified_message['images'] = message['images']
                simp_msg_history.append(simplified_message)

            # Processing the chat
            output_combined = ""
            for chunk in stream_chat(simp_msg_history, model=model, temperature=temperature):
                output_combined += chunk
                await websocket.send_json({"status": "generating", "response": output_combined, 'completed': False})

            # Fetch semantic category from the output
            semantic_compression = SemanticCompression(model=f"ollama:{model}", api_key=get_openai_api_key())
            semantic_category = semantic_compression.fetch_semantic_category(output_combined)

            # Send the final completed message
            await websocket.send_json(
                {"status": "completed", "response": output_combined, "semantic_category": semantic_category, "completed": True})

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})
        await websocket.close()


@router.websocket("/websocket_debate")
async def debate(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload["message"]
            message_history = payload["message_history"]
            model = payload.get("model", "solar")
            temperature = float(payload.get("temperature", 0.04))
            current_topic = payload.get("topic", "Unknown")

            # Set system prompt
            has_topic = False
            
            if current_topic != "Unknown":
                has_topic = True
                prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is {current_topic}.\n"

            system_prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is unknown, so try not to make any judgements thus far - only re-express the input words in your own style:"
            user_prompt = ""
            if message_history:
                # Add the message history prior to the message
                user_prompt += '\n'.join(msg['role'] + ": " + msg['content'] for msg in message_history)

            print(f"\t[ system prompt :: {system_prompt} ]")
            print(f"\t[ user prompt :: {user_prompt} ]")
            simp_msg_history = [{'role': 'system', 'content': system_prompt}]

            # Simplify message history to required format
            for message in message_history:
                simplified_message = {'role': message['role'], 'content': message['content']}
                if 'images' in message:
                    simplified_message['images'] = message['images']
                simp_msg_history.append(simplified_message)

            # Processing the chat
            output_combined = ""
            for chunk in stream_chat(simp_msg_history, model=model, temperature=temperature):
                output_combined += chunk
                await websocket.send_json({"status": "generating", "response": output_combined, 'completed': False})

            # Fetch semantic category from the output
            semantic_compression = SemanticCompression(model=f"ollama:{model}", api_key=get_openai_api_key())
            semantic_category = semantic_compression.fetch_semantic_category(output_combined)

            # Send the final completed message
            await websocket.send_json(
                {"status": "completed", "response": output_combined, "semantic_category": semantic_category, "completed": True})

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})
        await websocket.close()
