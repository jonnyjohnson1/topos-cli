# topos/channel/debatesim.py

import os
from uuid import uuid4

from dotenv import load_dotenv

import json
from datetime import datetime
import time
from fastapi import WebSocket, WebSocketDisconnect
from topos.FC.semantic_compression import SemanticCompression
from ..config import get_openai_api_key
from ..models.llm_classes import vision_models
from ..generations.ollama_chat import stream_chat
from ..utilities.utils import create_conversation_string
from ..services.classification_service.base_analysis import base_text_classifier, base_token_classifier
from topos.FC.conversation_cache_manager import ConversationCacheManager
from topos.FC.semantic_compression import SemanticCompression
from topos.FC.ontological_feature_detection import OntologicalFeatureDetection

# chess is more complicated than checkers but less complicated than go

# current:
# graph LR
#     timestamp["Timestamp: 2024-06-08T23:47:36.059626"]
#     user["user (USER)"]
#     sessionTEMP["sessionTEMP (SESSION)"]
#     userPRIME["userPRIME (USER)"]
#     than --> checkers
#     sessionTEMP --> of
#     checkers --> complicated
#     message --> is
#     userPRIME --> for
#     is --> chess
#     is --> message
#     checkers --> than
#     of --> sessionTEMP
#     chess --> is
#     for --> userPRIME
#     complicated --> is
#     timestamp --> user

# target:
# graph LR
#     userPRIME["userPRIME (USER)"]
#     sessionTEMP["sessionTEMP (SESSION)"]
#     timestamp["Timestamp: 2024-06-08T23:18:05.206590"]
#     message["message"]
#     chess["chess"]
#     more_complicated["more complicated"]
#     checkers["checkers"]
#     less_complicated["less complicated"]
#     go["go"]
#
#     userPRIME --> user
#     sessionTEMP --> session
#     timestamp --> user
#     message --> userPRIME
#     message --> sessionTEMP
#     message --> timestamp
#     chess --> message
#     more_complicated --> chess
#     more_complicated --> checkers
#     less_complicated --> chess
#     less_complicated --> go


class DebateSimulator:
    def __init__(self):
        load_dotenv()  # Load environment variables

        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.showroom_db_name = os.getenv("NEO4J_SHOWROOM_DATABASE")

        # self.cache_manager = ConversationCacheManager()
        self.ontological_feature_detection = OntologicalFeatureDetection(neo4j_uri, neo4j_user, neo4j_password,
                                                                         self.showroom_db_name)

    def get_ontology(self, user_id, session_id, message_id, message):
        composable_string = f"for user {user_id}, of {session_id}, the message is: {message}"
        print(f"\t\t[ composable_string :: {composable_string} ]")

        entities, pos_tags, dependencies, relations, srl_results, timestamp, context_entities = self.ontological_feature_detection.build_ontology_from_paragraph(
            user_id, session_id, message_id, composable_string)

        self.ontological_feature_detection.store_ontology(user_id, session_id, message_id, message, timestamp, context_entities, relations)

        input_components = message, entities, dependencies, relations, srl_results, timestamp, context_entities

        mermaid_syntax = self.ontological_feature_detection.extract_mermaid_syntax(input_components, input_type="components")
        return mermaid_syntax

    def search_messages_by_user(self, user_id):
        return self.ontological_feature_detection.get_messages_by_user(user_id)

    def search_messages_by_session(self, session_id):
        return self.ontological_feature_detection.get_messages_by_session(session_id)

    def search_users_by_session(self, session_id):
        return self.ontological_feature_detection.get_users_by_session(session_id)

    def search_sessions_by_user(self, user_id):
        return self.ontological_feature_detection.get_sessions_by_user(user_id)

    async def debate_step(self, websocket: WebSocket, data, app_state):
        payload = json.loads(data)
        message = payload["message"]

        # create a new message id, with 36 characters max
        message_id = str(uuid4())

        # check for collisions
        while self.ontological_feature_detection.check_message_exists(message_id):
            # re-roll a new message id, with 36 characters max
            message_id = str(uuid4())

        # session_id = payload["session_id"]
        # user_id = payload["user_id"]
        user_id = app_state.get("user_id", "")
        session_id = app_state.get("session_id", "")

        message_history = payload["message_history"]
        model = payload.get("model", "solar")
        temperature = float(payload.get("temperature", 0.04))
        current_topic = payload.get("topic", "Unknown")

        prior_ontology = app_state.get("prior_ontology", [])

        # if prior_ontology is []:
        #     prior_ontology = []

        current_ontology = self.get_ontology(user_id, session_id, message_id, message)

        mermaid_to_ascii = self.ontological_feature_detection.mermaid_to_ascii(current_ontology)

        print(f"[ prior_ontology: {prior_ontology} ]")

        print(f"[ current_ontology: {current_ontology} ]")

        print(f"[ mermaid_to_ascii: {mermaid_to_ascii} ]")

        # app_state.write_ontology(current_ontology)

        # break previous messages into ontology
        # cache ontology
        # ** use diffuser to spot differentials
        # *** map causal ontology back to spot reference point
        # break current message into ontology
        # ** BLEU score a 10x return on the ontology
        # read ontology + newest

        await self.think(topic="Chess vs Checkers", messages=message_history)

        # topic:
        # checkers is better than chess
        #
        # user1:
        # [user1] chess is better than checkers
        #
        # user2:
        # [user2] no, checkers is better than chess - it's faster
        #
        # user1:
        # [user1]  I don't believe so - checkers always takes at least a large fixed time to perform moves, and chess can mate in less than 10 if you're good
        #
        # user2:
        # [user2]  but checkers is more accessible to a wider audience, and it's easier to learn
        #
        # user1:
        # [user1]  that's true, but chess has a richer history and more complex strategies
        #
        # user2:
        # [user2]  but checkers is more fun to play, and it's more engaging

        # Set system prompt
        has_topic = False
        system_prompt = f""

        user_definition_prompt = f"""-----
                                    Users are defined by the following roles: user1, user2, user3, etc. The moderator is defined by the role: moderator.\n
                                    Roles are indicated by the format:
                                    [user1]: "I have an opinion on XYZ"
                                    [user2]: "I have another opinion on ABC"
                                    [moderator]: "I think the topic might be XYZ" 
                                    ------
                                    """

        if current_topic == "unknown topic":
            system_prompt = f"""You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is unknown, so try not to make any judgements thus far - only re-express the input words in your own style, in the format of:\n
                            {{\"role\":\"moderator\", \"content\":\"I think the topic might be...(_insert name of what you think the topic might be based on the ongoing discussion here!_)\", \"certainty_score\": \"(_insert certainty score 1-10 here!_)\"}}"""
        else:
            has_topic = True
            system_prompt = f"""You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is {current_topic}.\n
                            You keep track of who is speaking, in the context of saying out loud every round:\n
                            {{\"role\": \"moderator\", \"content\": \"The topic is...(_insert name of topic here!_)\", \"synopsis\": \"(_insert synopsis of the content so far, with debaters points in abstract_)\", \"affirmative_negative score\": \"(_insert debate affirmative (is affirming the premise of the current \'topic\') score, 1 to 10, here!_) / (_insert debate negative (is not affirming the premise of the current "topic", and is correlated to the inverse of the statement) score, 1 to 10, here!_)\"}}"""

        system_prompt = f"{user_definition_prompt}\n{system_prompt}"

        user_prompt = f""
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

        output_json = []
        try:
            result = json.loads(f"{output_combined}")
            output_json = result
        except json.JSONDecodeError:
            output_json = output_combined
            print(f"\t\t[ error in decoding :: {output_combined} ]")

        # Fetch semantic category from the output
        semantic_compression = SemanticCompression(model=f"ollama:{model}", api_key=get_openai_api_key())
        semantic_category = semantic_compression.fetch_semantic_category(output_combined)

        # Send the final completed message
        await websocket.send_json(
            {"status": "completed", "response": output_combined, "semantic_category": semantic_category,
             "completed": True})


    async def think(self, topic, messages):
        # Implement the think method logic here
        pass


#alright! once again, same style, same acumen, boil over each and every one of those

#okay compose it all in a series of functions so I can copy paste.

#AFTERWARDS I'd like a list of all of the new functions you need to yet provide super-stubs for