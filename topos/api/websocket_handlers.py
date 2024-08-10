import logging
import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
import json
from datetime import datetime
import time
import traceback
from dotenv import load_dotenv

# Import the chat model classes
from topos.FC.semantic_compression import SemanticCompression
from ..config import get_openai_api_key
from ..models.llm_classes import vision_models
from ..utilities.utils import create_conversation_string
from ..services.classification_service.base_analysis import base_text_classifier, base_token_classifier
from topos.FC.conversation_cache_manager import ConversationCacheManager
from topos.channel.debatesim import DebateSimulator

# Load environment variables
load_dotenv()

# Define the FORCE_OPENAI_MODE constant
FORCE_OPENAI_MODE = True


class WebsocketHandler:
    def __init__(self, model):
        self.model = model
        self.router = APIRouter()
        self.debate_simulator = DebateSimulator.get_instance()

        # Register WebSocket routes
        self.router.websocket("/websocket_chat")(self.chat)
        self.router.websocket("/websocket_meta_chat")(self.meta_chat)
        self.router.websocket("/websocket_chat_summary")(self.chat_summary)
        self.router.websocket("/websocket_mermaid_chart")(self.mermaid_chart)

    async def chat(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                await self.handle_chat(websocket, payload)
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            stack_trace = traceback.format_exc()
            await websocket.send_json({"status": "error", "message": str(e), "trace": stack_trace})
            await websocket.close()

    async def handle_chat(self, websocket: WebSocket, payload):
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

        # Update default_config with provided processing_config, if any
        config = {**default_config, **processing_config}

        # Set system prompt based on the topic
        system_prompt = self.generate_system_prompt(current_topic)

        # Simplify message history to required format
        simp_msg_history = [{'role': 'system', 'content': system_prompt}]
        for message in message_history:
            simplified_message = {'role': message['role'], 'content': message['content']}
            simp_msg_history.append(simplified_message)

        last_message = simp_msg_history[-1]['content']
        role = simp_msg_history[-1]['role']

        # Fetch base, per-message token classifiers
        if config['calculateInMessageNER']:
            start_time = time.time()
            base_analysis = base_token_classifier(last_message)  # this is only an ner dict atm
            duration = time.time() - start_time
            print(f"\t[ base_token_classifier duration: {duration:.4f} seconds ]")

        # Fetch base, per-message text classifiers
        if config['calculateModerationTags']:
            start_time = time.time()
            text_classifiers = {}
            try:
                text_classifiers = base_text_classifier(last_message)
            except Exception as e:
                print(f"Failed to compute base_text_classifier: {e}")
            duration = time.time() - start_time
            print(f"\t[ base_text_classifier duration: {duration:.4f} seconds ]")

        conv_cache_manager = ConversationCacheManager()
        if config['calculateModerationTags'] or config['calculateInMessageNER']:
            print(f"\t[ save to conv cache :: conversation {conversation_id}-{message_id} ]")
            try:
                dummy_data = {
                    message_id:
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
            await websocket.send_json({"status": "fetched_user_analysis", 'user_message': dummy_data})
        else:
            print(f"\t[ save to conv cache :: conversation {conversation_id}-{message_id} ]")
            # Saving an empty dictionary for the message id
            conv_cache_manager.save_to_cache(conversation_id, {
                message_id:
                    {
                        'role': role,
                        'message': last_message,
                        'timestamp': datetime.now(),
                    }})

        # Processing the chat
        output_combined = ""
        for chunk in self.model.stream_chat(simp_msg_history, temperature=temperature):
            output_combined += chunk
            await websocket.send_json({"status": "generating", "response": output_combined, 'completed': False})

        # Fetch semantic category from the output
        semantic_compression = SemanticCompression(model=f"ollama:{self.model.model_name}",
                                                   api_key=get_openai_api_key())
        semantic_category = (semantic_compression.fetch_semantic_category(output_combined)).content

        # Start timer for base_token_classifier
        if config['calculateInMessageNER']:
            start_time = time.time()
            base_analysis = base_token_classifier(output_combined)
            duration = time.time() - start_time
            print(f"\t[ base_token_classifier duration: {duration:.4f} seconds ]")

        # Start timer for base_text_classifier
        if config['calculateModerationTags']:
            start_time = time.time()
            text_classifiers = base_text_classifier(output_combined)
            duration = time.time() - start_time
            print(f"\t[ base_text_classifier duration: {duration:.4f} seconds ]")

        if config['calculateModerationTags'] or config['calculateInMessageNER']:
            print(f"\t[ save to conv cache :: conversation {conversation_id}-{chatbot_msg_id} ]")
            dummy_bot_data = {
                chatbot_msg_id:
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
        else:
            # Saving an empty dictionary for the message id
            print(f"\t[ save to conv cache :: conversation {conversation_id}-{chatbot_msg_id} ]")
            conv_cache_manager.save_to_cache(conversation_id, {
                chatbot_msg_id:
                    {
                        'role': "ChatBot",
                        'message': output_combined,
                        'timestamp': datetime.now(),
                    }})

        # Send the final completed message
        send_pkg = {"status": "completed", "response": output_combined, "semantic_category": semantic_category,
                    "completed": True}
        if config['calculateModerationTags'] or config['calculateInMessageNER']:
            send_pkg['user_message'] = dummy_data
            send_pkg['bot_data'] = dummy_bot_data

        await websocket.send_json(send_pkg)

    async def meta_chat(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                await self.handle_meta_chat(websocket, payload)
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            await websocket.send_json({"status": "error", "message": str(e)})
            await websocket.close()

    async def handle_meta_chat(self, websocket: WebSocket, payload):
        message = payload["message"]
        message_history = payload["message_history"]
        meta_conv_message_history = payload["meta_conv_message_history"]
        temperature = float(payload.get("temperature", 0.04))
        current_topic = payload.get("topic", "Unknown")

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
        for chunk in self.model.stream_chat(simp_msg_history, temperature=temperature):
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

    async def chat_summary(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                await self.handle_chat_summary(websocket, payload)
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            await websocket.send_json({"status": "error", "message": str(e)})
            await websocket.close()

    async def handle_chat_summary(self, websocket: WebSocket, payload):
        conversation_id = payload["conversation_id"]
        subject = payload.get("subject", "knowledge")
        temperature = float(payload.get("temperature", 0.04))

        # load conversation
        cache_manager = ConversationCacheManager()
        conv_data = cache_manager.load_from_cache(conversation_id)
        if conv_data is None:
            raise HTTPException(status_code=404, detail="Conversation not found in cache")

        context = create_conversation_string(conv_data, 12)

        print(f"\t[ generating summary :: model {self.model.model_name} :: subject {subject}]")

        # Set system prompt
        system_prompt = "PRESENT CONVERSATION:\n-------<context>" + context + "\n-------\n"
        query = f"""Summarize this conversation. Frame your response around the subject of {subject}"""

        msg_history = [{'role': 'system', 'content': system_prompt}]

        # Append the present message to the message history
        simplified_message = {'role': "user", 'content': query}
        msg_history.append(simplified_message)

        # Processing the chat
        output_combined = ""
        for chunk in self.model.stream_chat(msg_history, temperature=temperature):
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

    async def mermaid_chart(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                await self.handle_mermaid_chart(websocket, payload)
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            await websocket.send_json({"status": "error", "message": str(e)})
            await websocket.close()

    async def handle_mermaid_chart(self, websocket: WebSocket, payload):
        conversation_id = payload["conversation_id"]
        subject = payload.get("subject", "knowledge")
        temperature = float(payload.get("temperature", 0.04))

        # load conversation
        cache_manager = ConversationCacheManager()
        conv_data = cache_manager.load_from_cache(conversation_id)
        if conv_data is None:
            raise HTTPException(status_code=404, detail="Conversation not found in cache")

        context = create_conversation_string(conv_data, 12)

        print(f"\t[ generating mermaid chart :: model {self.model.model_name} :: subject {subject}]")

        # Set system prompt
        system_prompt = "PRESENT CONVERSATION:\n-------<context>" + context + "\n-------\n"
        query = f"""Create a Mermaid chart that visualizes the flow and key points of this conversation. Focus on the subject of {subject}. Use appropriate Mermaid syntax."""

        msg_history = [{'role': 'system', 'content': system_prompt}]

        # Append the present message to the message history
        simplified_message = {'role': "user", 'content': query}
        msg_history.append(simplified_message)

        # Processing the chat
        output_combined = ""
        for chunk in self.model.stream_chat(msg_history, temperature=temperature):
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

    def generate_system_prompt(self, topic: str) -> str:
        return f"""You are an AI assistant engaged in a conversation. The current topic is: {topic}. 
        Provide informative and engaging responses while staying on topic. If asked about events after April 2024, 
        respond as a highly informed individual from April 2024 would, acknowledging the limitation when relevant."""
