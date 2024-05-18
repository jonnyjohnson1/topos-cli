from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..generations.ollama_chat import stream_chat
from topos.FC.semantic_compression import SemanticCompression
from ..config import get_openai_api_key
from ..models.llm_classes import vision_models
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
            # If user uses a vision model, load images, else don't
            isVisionModel = model in vision_models
            print(f"\t[ using model :: {model} :: 🕶️  isVision ]") if isVisionModel else print(f"\t[ using model :: {model} ]")  
            
            for message in message_history:
                simplified_message = {'role': message['role'], 'content': message['content']}
                if 'images' in message and isVisionModel:
                    simplified_message['images'] = message['images']
                simp_msg_history.append(simplified_message)
            
            # Check 

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
            system_prompt = f""
            
            if current_topic == "unknown topic":
                system_prompt = (f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is unknown, so try not to make any judgements thus far - only re-express the input words in your own style, in the format of:\n"
                                 f"{{\"role\":\"moderator\", \"content\":\"I think the topic might be...(_insert name of what you think the topic might be based on the ongoing discussion here!_)\", \"certainty_score\": \"(_insert certainty score 1-10 here!_)\"}}")
            else:
                has_topic = True
                system_prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is {current_topic}.\n"
                system_prompt += (f"You keep track of who is speaking, in the context of saying out loud every round:\n"
                                  f"{{\"role\":\"moderator\", \"content\":\"The topic is...(_insert name of topic here!_)\"}}")

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

            # test = {'role': 'moderator', 'content': 'I think the topic might be Chess versus Checkers.', 'certainty_score': '8'}
            # test_msg = json.dumps(test)

            output_json = []
            try:
                result = json.loads(f"{output_combined}")
                output_json = result
            except json.JSONDecodeError:
                output_json = output_combined
                print(f"\t\t[ error in decoding :: {output_combined} ]")

            # user1: i want to debate chess vs checkers                                           user2: i want to debate chess vs go                            moderator insight: maybe the debate is, in and of itself, at a higher level of abstraction - maybe we should debate about a. debate checkers vs chess, vs. b debate checkers vs go?

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
