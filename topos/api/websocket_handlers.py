from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..generations.ollama_chat import stream_chat
from topos.FC.semantic_compression import SemanticCompression
from ..config import get_openai_api_key
from ..models.llm_classes import vision_models
import json

from ..services.classification_service.base_analysis import base_text_classifier, base_token_classifier
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
            # If user uses a vision model, load images, else don't
            isVisionModel = model in vision_models
            print(f"\t[ using model :: {model} :: 🕶️  isVision ]") if isVisionModel else print(f"\t[ using model :: {model} ]")  
            
            for message in message_history:
                simplified_message = {'role': message['role'], 'content': message['content']}
                if 'images' in message and isVisionModel:
                    simplified_message['images'] = message['images']
                simp_msg_history.append(simplified_message)
            
            # Fetch base, per-message token classifiers
            last_message = simp_msg_history[-1]['content']
            print("Last message: " + last_message)
            entity_dict = base_token_classifier(last_message)
            if len(entity_dict) > 0:
                print("ENTS (ner) :: ", entity_dict)

            # Fetch base, per-message text classifiers
            text_classifiers = base_text_classifier(last_message)
            print("Text Classifiers :: ", entity_dict)

            # Processing the chat
            output_combined = ""
            for chunk in stream_chat(simp_msg_history, model=model, temperature=temperature):
                output_combined += chunk
                await websocket.send_json({"status": "generating", "response": output_combined, 'completed': False})

            # Fetch semantic category from the output
            semantic_compression = SemanticCompression(model=f"ollama:{model}", api_key=get_openai_api_key())
            semantic_category = semantic_compression.fetch_semantic_category(output_combined)

            # Fetch base, per-message token classifiers
            last_message = simp_msg_history[-1]['content']
            print("system msg :: " + output_combined)
            entity_dict = base_token_classifier(output_combined)
            if len(entity_dict) > 0:
                print("ENTS (ner) :: ", entity_dict)

            # Fetch base, per-message text classifiers
            text_classifiers = base_text_classifier(output_combined)
            print("Text Classifiers :: ", text_classifiers)

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
