from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from datetime import datetime
import time
import traceback
import pprint

from ..generations.chat_gens import LLMChatGens
# from topos.FC.semantic_compression import SemanticCompression
# from ..config import get_openai_api_key
from ..models.llm_classes import vision_models
import json

from ..utilities.utils import create_conversation_string
from ..services.classification_service.base_analysis import base_text_classifier, base_token_classifier
from ..services.loggers.process_logger import ProcessLogger
from ..services.ontology_service.mermaid_chart import MermaidCreator

# cache database
from topos.FC.conversation_cache_manager import ConversationCacheManager

# Debate simulator
from topos.channel.debatesim import DebateSimulator

router = APIRouter()
debate_simulator = DebateSimulator.get_instance()

async def end_ws_process(websocket, websocket_process, process_logger, send_json, write_logs=True):
    await process_logger.end(websocket_process)
    if write_logs:
        logs = process_logger.get_logs()
        pprint.pp(logs)
        # for step_name, log_data in logs.items():
        #     details = '|'.join([f"{key}={value}" for key, value in log_data.get("details", {}).items()])
        #     log_message = (
        #         f"{step_name},{process_logger.step_id},"
        #         f"{log_data['start_time']},{log_data.get('end_time', '')},"
        #         f"{log_data.get('elapsed_time', '')},{details}"
        #     )
            # await process_logger.log(log_message) # available when logger client is made
    await websocket.send_json(send_json)

@router.websocket("/websocket_chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    process_logger = ProcessLogger(verbose=False, run_logger=False)
    websocket_process = "/websocket_chat"
    await process_logger.start(websocket_process)
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            conversation_id = payload["conversation_id"]
            message_id = payload["message_id"]
            chatbot_msg_id = payload["chatbot_msg_id"]
            message = payload["message"]
            message_history = payload["message_history"]
            temperature = float(payload.get("temperature", 0.04))
            current_topic = payload.get("topic", "Unknown")
            processing_config = payload.get("processing_config", {})
            
           
            # Set default values if any key is missing or if processing_config is None
            default_config = {
                "showInMessageNER": True,
                "calculateInMessageNER": True,
                "showModerationTags": True,
                "calculateModerationTags": True,
                "showSidebarBaseAnalytics": True
            }
            
            # model specifications
            model = payload.get("model", "solar")
            provider = payload.get('provider', 'ollama') # defaults to ollama right now
            api_key = payload.get('api_key', 'ollama')

            llm_client = LLMChatGens(model_name=model, provider=provider, api_key=api_key)


            # Update default_config with provided processing_config, if any
            config = {**default_config, **processing_config}

            # Set system prompt
            has_topic = False
            
            if current_topic != "Unknown":
                has_topic = True
                prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is {current_topic}.\n"

            system_prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator."
            user_prompt = ""
            if message_history:
                # Add the message history prior to the message
                user_prompt += '\n'.join(msg['role'] + ": " + msg['content'] for msg in message_history)

            # print(f"\t[ system prompt :: {system_prompt} ]")
            # print(f"\t[ user prompt :: {user_prompt} ]")
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
            
            last_message = simp_msg_history[-1]['content']
            role = simp_msg_history[-1]['role']
            num_user_toks = len(last_message.split())
            # Fetch base, per-message token classifiers
            if config['calculateInMessageNER']:
                await process_logger.start("calculateInMessageNER-user", num_toks=num_user_toks)
                start_time = time.time()
                base_analysis = base_token_classifier(last_message)  # this is only an ner dict atm
                duration = time.time() - start_time
                await process_logger.end("calculateInMessageNER-user")
                print(f"\t[ base_token_classifier duration: {duration:.4f} seconds ]")
            
            # Fetch base, per-message text classifiers
            # Start timer for base_text_classifier
            if config['calculateModerationTags']:
                await process_logger.start("calculateModerationTags-user", num_toks=num_user_toks)
                start_time = time.time()
                text_classifiers = {}
                try:
                    text_classifiers = base_text_classifier(last_message)
                except Exception as e:
                    print(f"Failed to compute base_text_classifier: {e}")
                duration = time.time() - start_time
                await process_logger.end("calculateModerationTags-user")
                print(f"\t[ base_text_classifier duration: {duration:.4f} seconds ]")
            
            conv_cache_manager = ConversationCacheManager()
            if config['calculateModerationTags'] or config['calculateInMessageNER']:
                await process_logger.start("saveToConversationCache-user")
                print(f"\t[ save to conv cache :: conversation {conversation_id}-{message_id} ]")
                try:
                    dummy_data = {
                        message_id : 
                            {
                            'role': role,
                            'timestamp': datetime.now(), 
                            'message': last_message
                        }}
                except Exception as e:
                    print("Error", e)
                if config['calculateInMessageNER']:
                    dummy_data[message_id]['in_line'] = {'base_analysis': base_analysis}
                if config['calculateModerationTags']:
                    dummy_data[message_id]['commenter'] = {'base_analysis': text_classifiers}

                conv_cache_manager.save_to_cache(conversation_id, dummy_data)
                # Removing the keys from the nested dictionary
                if message_id in dummy_data:
                    dummy_data[message_id].pop('message', None)
                    dummy_data[message_id].pop('timestamp', None)
                # Sending first batch of user message analysis back to the UI
                await process_logger.end("saveToConversationCache-user")
                await websocket.send_json({"status": "fetched_user_analysis", 'user_message': dummy_data})
            else:
                print(f"\t[ save to conv cache :: conversation {conversation_id}-{message_id} ]")
                await process_logger.start("saveToConversationCache-user")
                # Saving an empty dictionary for the messag id
                conv_cache_manager.save_to_cache(conversation_id, {
                    message_id : 
                        {
                        'role': role,
                        'message': last_message, 
                        'timestamp': datetime.now(), 
                    }})
                await process_logger.end("saveToConversationCache-user")

            # Processing the chat
            output_combined = ""
            is_first_token = True
            total_tokens = 0  # Initialize token counter
            ttfs = 0 # init time to first token value
            await process_logger.start("llm_generation_stream_chat", provider=provider, model=model, len_msg_hist=len(simp_msg_history))
            start_time = time.time()  # Track the start time for the whole process
            for chunk in llm_client.stream_chat(simp_msg_history, temperature=temperature):
                if len(chunk) > 0:
                    if is_first_token:
                        ttfs_end_time = time.time()
                        ttfs = ttfs_end_time - start_time
                        is_first_token = False
                    output_combined += chunk
                    total_tokens += len(chunk.split())
                    await websocket.send_json({"status": "generating", "response": output_combined, 'completed': False})
            end_time = time.time()  # Capture the end time
            elapsed_time = end_time - start_time  # Calculate the total elapsed time
            # Calculate tokens per second
            if elapsed_time > 0:
                tokens_per_second = total_tokens / elapsed_time
            ttl_num_toks = 0
            for i in simp_msg_history:
                if isinstance(i['content'], str):
                    ttl_num_toks += len(i['content'].split())
            await process_logger.end("llm_generation_stream_chat", toks_per_sec=f"{tokens_per_second:.1f}", ttfs=f"{ttfs}", num_toks=num_user_toks, ttl_num_toks=ttl_num_toks)
            # Fetch semantic category from the output
            # semantic_compression = SemanticCompression(model=f"ollama:{model}", api_key=get_openai_api_key())
            # semantic_category = semantic_compression.fetch_semantic_category(output_combined)

            num_response_toks=len(output_combined.split())
            # Start timer for base_token_classifier
            if config['calculateInMessageNER']:
                await process_logger.start("calculateInMessageNER-ChatBot", num_toks=num_response_toks)
                start_time = time.time()
                base_analysis = base_token_classifier(output_combined)
                duration = time.time() - start_time
                print(f"\t[ base_token_classifier duration: {duration:.4f} seconds ]")
                await process_logger.end("calculateInMessageNER-ChatBot")

            # Start timer for base_text_classifier
            if config['calculateModerationTags']:
                await process_logger.start("calculateModerationTags-ChatBot", num_toks=num_response_toks)
                start_time = time.time()
                text_classifiers = base_text_classifier(output_combined)
                duration = time.time() - start_time
                print(f"\t[ base_text_classifier duration: {duration:.4f} seconds ]")
                await process_logger.end("calculateModerationTags-ChatBot")

            if config['calculateModerationTags'] or config['calculateInMessageNER']:
                await process_logger.start("saveToConversationCache-ChatBot")
                print(f"\t[ save to conv cache :: conversation {conversation_id}-{chatbot_msg_id} ]")
                dummy_bot_data = {
                    chatbot_msg_id : 
                        {
                        'role': "ChatBot",
                        'message': output_combined, 
                        'timestamp': datetime.now(), 
                    }}
                if config['calculateInMessageNER']:
                    dummy_bot_data[chatbot_msg_id]['in_line'] = {'base_analysis': base_analysis}
                if config['calculateModerationTags']:
                    dummy_bot_data[chatbot_msg_id]['commenter'] = {'base_analysis': text_classifiers}
                conv_cache_manager.save_to_cache(conversation_id, dummy_bot_data)
                # Removing the keys from the nested dictionary
                if chatbot_msg_id in dummy_bot_data:
                    dummy_bot_data[chatbot_msg_id].pop('message', None)
                    dummy_bot_data[chatbot_msg_id].pop('timestamp', None)
                await process_logger.end("saveToConversationCache-ChatBot")
            else:
                # Saving an empty dictionary for the messag id
                print(f"\t[ save to conv cache :: conversation {conversation_id}-{chatbot_msg_id} ]")
                await process_logger.start("saveToConversationCache-ChatBot")
                conv_cache_manager.save_to_cache(conversation_id, {
                    chatbot_msg_id : 
                        {
                        'role': "ChatBot",
                        'message': output_combined, 
                        'timestamp': datetime.now(), 
                    }})
                await process_logger.end("saveToConversationCache-ChatBot")
            
            # Send the final completed message
            send_pkg = {"status": "completed", "response": output_combined, "completed": True}
            if config['calculateModerationTags'] or config['calculateInMessageNER']:
                send_pkg['user_message'] = dummy_data
                send_pkg['bot_data'] = dummy_bot_data
            
            await end_ws_process(websocket, websocket_process, process_logger, send_pkg)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})
        await websocket.close()

@router.websocket("/websocket_meta_chat")
async def meta_chat(websocket: WebSocket):
    """
    A chat about conversations.
    This conversation is geared towards exploring the different directions
    a speaker wishes to engage with a chat.
    How to present themselves with _______ (personality, to elicit responses)
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload["message"]
            message_history = payload["message_history"]
            meta_conv_message_history = payload["meta_conv_message_history"]
            model = payload.get("model", "solar")
            temperature = float(payload.get("temperature", 0.04))
            current_topic = payload.get("topic", "Unknown")

            
            # model specifications
            model = payload.get("model", "solar")
            provider = payload.get('provider', 'ollama') # defaults to ollama right now
            api_key = payload.get('api_key', 'ollama')

            llm_client = LLMChatGens(model_name=model, provider=provider, api_key=api_key)

            # Set system prompt
            system_prompt = f"""You are a highly skilled conversationalist, adept at communicating strategies and tactics. Help the user navigate their current conversation to determine what to say next. 
            You possess a private, unmentioned expertise: PhDs in CBT and DBT, an elegant, smart, provocative speech style, extensive world travel, and deep literary theory knowledge à la Terry Eagleton. Demonstrate your expertise through your guidance, without directly stating it."""
            
            print(f"\t[ system prompt :: {system_prompt} ]")
            
            # Add the actual chat to the system prompt
            if len(message_history) > 0:
                system_prompt += f"\nThe conversation thus far has been this:\n-------\n"
                if message_history:
                    # Add the message history prior to the message
                    system_prompt += '\n'.join(msg['role'] + ": " + msg['content'] for msg in message_history)
                    system_prompt += '\n-------'

            simp_msg_history = [{'role': 'system', 'content': system_prompt}]

            # Simplify message history to required format
            for message in meta_conv_message_history:
                simplified_message = {'role': message['role'], 'content': message['content']}
                if 'images' in message:
                    simplified_message['images'] = message['images']
                simp_msg_history.append(simplified_message)

            # Processing the chat
            output_combined = ""
            for chunk in llm_client.stream_chat(simp_msg_history, temperature=temperature):
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

            llm_client = LLMChatGens(model_name=model, provider=provider, api_key=api_key)


            # load conversation
            cache_manager = ConversationCacheManager()
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

            llm_client = LLMChatGens(model_name=model, provider=provider, api_key=api_key)

            mermaid_generator = MermaidCreator(llm_client)
            # load conversation
            if full_conversation:
                cache_manager = ConversationCacheManager()
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




@router.websocket("/debate_flow_with_jwt")
async def debate_flow_with_jwt(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message_data = payload.get("message_data", None)
            model = payload.get("model", None)
            
            if message_data:
                await websocket.send_json({"status": "generating", "response": "starting debate flow analysis", 'completed': False})
                try:
                    # Assuming DebateSimulator is correctly set up
                    debate_simulator = await DebateSimulator.get_instance()
                    response_data = debate_simulator.process_messages(message_data, model)
                    await websocket.send_json({"status": "completed", "response": response_data, 'completed': True})
                except Exception as e:
                    await websocket.send_json({"status": "error", "response": f"Error: {e}", 'completed': True})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})
        await websocket.close()
    finally:
        await websocket.close()