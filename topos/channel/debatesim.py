# topos/channel/debatesim.py
import hashlib

from typing import Dict, List

import os
import uuid
from uuid import uuid4
import threading
from queue import Queue

from dotenv import load_dotenv

import jwt
from jwt.exceptions import InvalidTokenError

import json
from datetime import datetime
import time

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import entropy

from fastapi import WebSocket, WebSocketDisconnect
from ..FC.argument_detection import ArgumentDetection
from ..config import get_openai_api_key
from ..models.llm_classes import vision_models
from ..generations.ollama_chat import stream_chat
from ..services.database.app_state import AppState
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
    _instance = None
    _lock = threading.Lock()

    @staticmethod
    def get_instance():
        if DebateSimulator._instance is None:
            with DebateSimulator._lock:
                if DebateSimulator._instance is None:
                    DebateSimulator._instance = DebateSimulator()
        return DebateSimulator._instance

    def __init__(self, use_neo4j=False):
        if DebateSimulator._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            # Load the pre-trained model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')

            self.operational_llm_model = "ollama:dolphin-llama3"

            # Initialize the SentenceTransformer model for embedding text
            self.fast_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.argument_detection = ArgumentDetection(model=self.operational_llm_model, api_key="ollama")

            self.semantic_compression = SemanticCompression(model=self.operational_llm_model, api_key="ollama")

            self.app_state = AppState.get_instance()

            load_dotenv()  # Load environment variables

            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USER")
            neo4j_password = os.getenv("NEO4J_PASSWORD")
            self.showroom_db_name = os.getenv("NEO4J_SHOWROOM_DATABASE")
            self.use_neo4j = use_neo4j

            # self.cache_manager = ConversationCacheManager()
            self.ontological_feature_detection = OntologicalFeatureDetection(neo4j_uri, neo4j_user, neo4j_password,
                                                                             self.showroom_db_name, self.use_neo4j)

            # JWT secret key (should be securely stored, e.g., in environment variables)
            self.jwt_secret = os.getenv("JWT_SECRET")

            self.task_queue = Queue()
            self.processing_thread = threading.Thread(target=self.process_tasks)
            self.processing_thread.daemon = True
            self.processing_thread.start()

    def add_task(self, task):
        self.task_queue.put(task)

    def process_tasks(self):
        while True:
            task = self.task_queue.get()
            if task['type'] == 'reset':
                self.reset_processing_queue()
            else:
                self.execute_task(task)
            self.task_queue.task_done()

    def reset_processing_queue(self):
        # Logic to reset the processing queue
        while not self.task_queue.empty():
            self.task_queue.get()
            self.task_queue.task_done()
        print("Processing queue has been reset.")

    def execute_task(self, task):
        # Process the task based on its type and data
        if task['type'] == 'check_and_reflect':
            self.check_and_reflect(task['session_id'], task['user_id'], task['generation_nonce'], task['message_id'], task['message'])
        elif task['type'] == 'broadcast':
            self.start_broadcast_subprocess(task['websocket'], task['message'])
        # Add other task types as needed
        print(f"Executed task: {task['type']}")

    def websocket_broadcast(self, websocket, message):
        while True:
            if message:  # Condition to broadcast
                websocket.send(message)
            time.sleep(1)  # Adjust as necessary

    # Function to start the subprocess
    def start_broadcast_subprocess(self, websocket, message):
        broadcast_thread = threading.Thread(target=self.websocket_broadcast, args=(websocket, message))
        broadcast_thread.start()

    def get_ontology(self, user_id, session_id, message_id, message):
        composable_string = f"for user {user_id}, of {session_id}, the message is: {message}"
        print(f"\t\t[ composable_string :: {composable_string} ]")

        entities, pos_tags, dependencies, relations, srl_results, timestamp, context_entities = self.ontological_feature_detection.build_ontology_from_paragraph(
            user_id, session_id, message_id, composable_string)

        if self.use_neo4j:
            self.ontological_feature_detection.store_ontology(user_id, session_id, message_id, message, timestamp, context_entities, relations)

        input_components = message, entities, dependencies, relations, srl_results, timestamp, context_entities

        mermaid_syntax = self.ontological_feature_detection.extract_mermaid_syntax(input_components, input_type="components")
        return mermaid_syntax

    def search_messages_by_user(self, user_id):
        return self.ontological_feature_detection.get_messages_by_user(user_id)

    def search_messages_by_session(self, session_id, relation_type):
        return self.ontological_feature_detection.get_messages_by_session(session_id, relation_type)

    def search_users_by_session(self, session_id, relation_type):
        return self.ontological_feature_detection.get_users_by_session(session_id, relation_type)

    def search_sessions_by_user(self, user_id, relation_type):
        return self.ontological_feature_detection.get_sessions_by_user(user_id, relation_type)

    def has_message_id(self, message_id):
        if self.use_neo4j:
            return self.ontological_feature_detection.check_message_exists(message_id)
        else:
            return False

    # @note: integrate is post, due to constant
    async def integrate(self, token, data, app_state):
        payload = json.loads(data)
        message = payload["message"]

        # create a new message id, with 36 characters max
        message_id = str(uuid4())

        # check for collisions
        while self.has_message_id(message_id):
            # re-roll a new message id, with 36 characters max
            message_id = str(uuid4())

        # Decode JWT token to extract user_id and session_id
        try:
            decoded_token = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            user_id = decoded_token.get("user_id", "")
            session_id = decoded_token.get("session_id", "")
        except InvalidTokenError:
            # await websocket.send_json({"status": "error", "response": "Invalid JWT token"})
            return

        # if no user_id, bail
        if user_id == "" or session_id == "":
            return

        current_topic = payload.get("topic", "Unknown")

        # from app state
        message_history = app_state.get_value(f"message_history_{session_id}", [])

        prior_ontology = app_state.get_value(f"prior_ontology_{session_id}", [])

        current_ontology = self.get_ontology(user_id, session_id, message_id, message)

        print(f"[ prior_ontology: {prior_ontology} ]")
        print(f"[ current_ontology: {current_ontology} ]")

        prior_ontology.append(current_ontology)

        app_state.set_state(f"prior_ontology{session_id}_", prior_ontology)

        mermaid_to_ascii = self.ontological_feature_detection.mermaid_to_ascii(current_ontology)
        print(f"[ mermaid_to_ascii: {mermaid_to_ascii} ]")

        message_history.append(message)

        app_state.set_value(f"message_history_{session_id}", message_history)

        # Create new Generation
        generation_nonce = self.generate_nonce()

        self.add_task({
            'type': 'check_and_reflect',
            'session_id': session_id,
            'user_id': user_id,
            'generation_nonce': generation_nonce,
            'message_id': message_id,
            'message': message}
        )

        return current_ontology

    @staticmethod
    def generate_nonce():
        return str(uuid.uuid4())

    @staticmethod
    def aggregate_user_messages(message_history: List[Dict]) -> Dict[str, List[str]]:
        user_messages = {}
        for message in message_history:
            user_id = message['data']['user_id']
            content = message['data']['content']
            if user_id not in user_messages:
                user_messages[user_id] = []
            user_messages[user_id].append(content)
        return user_messages

    @staticmethod
    def generate_hash(cluster: List[str]) -> str:
        sorted_cluster = sorted(cluster)
        return hashlib.sha256(json.dumps(sorted_cluster).encode()).hexdigest()

    def generate_and_check_hashes(self, clusters):
        cluster_hashes = {}
        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            cluster_hash = self.generate_hash(cluster["messages"])
            cluster_hashes[cluster_id] = cluster_hash
        return cluster_hashes

    def incremental_clustering(self, clusters, cluster_hashes):
        updated_clusters = {}
        for cluster_id, cluster in clusters.items():
            if cluster_hashes[cluster_id] != self.generate_hash(cluster):
                updated_clusters[cluster_id] = cluster
        return updated_clusters

    async def broadcast_to_websocket_group(self, websocket_group, json_message):
        for websocket in websocket_group:
            await websocket.send_json(json_message)

    async def check_and_reflect(self, session_id, user_id, generation_nonce, generation_message_id):
        app_state = AppState().get_instance()

        message_history = app_state.get_value(f"message_history_{session_id}", [])

        # Step 1: Gather message history for specific users
        user_messages = self.aggregate_user_messages(message_history)
        print(f"\t[ reflect :: user_messages :: {user_messages} ]")

        # Step 2: Cluster analysis for each user's messages
        # Generate and check hashes for clusters
        clusters = self.cluster_messages(user_messages)
        print(f"\t[ reflect :: clustered_messages :: {clusters} ]")

        # @note:@here:@todo:@next: this should be linear, step 3 replace match and then keep pumping out websocket
        #  messages

        # Because clusters are based on sentences, groups of sentences will hash to the exact same value & will thus
        # have identical WEPCC parameters.
        cluster_hashes = self.generate_and_check_hashes(clusters)

        websocket_group = app_state.get_value(f"websocket_group_{session_id}", [])

        # Send initial cluster data back to frontend
        await self.broadcast_to_websocket_group(websocket_group, {
            "status": "initial_clusters",
            "clusters": clusters,
            "hashes": cluster_hashes,
            "generation": generation_nonce
        })

        # Perform incremental clustering if needed
        updated_clusters = self.incremental_clustering(clusters, cluster_hashes)
        # Send updated cluster data back to frontend
        await self.broadcast_to_websocket_group(websocket_group, {
            "status": "updated_clusters",
            "clusters": updated_clusters,
            "hashes": cluster_hashes,
            "generation": generation_nonce
        })

        # cluster message callback
        # each cluster is defined by a cluster id (a hash of its messages, messages sorted alphabetically)

        # 1. early out if the cluster is identical
        # 2. total message completion is based on all messages (a generation)
        # 3. previous generations DO NOT complete - they are halted upon a new message
        # 4. clustering will not be affected by other Users if their message has not changed, but generations
        #    always will because every new message from another player is dealt with re: claims/counterclaims
        # 5. technically, each generation has a final score (though because of processing reqs, we're not expecting
        #    to have more than each generation w/ a final score, technically this can be done as well, but
        #    it's probably not germane to the convo needs, so let's just not)

        # prioritize wepcc (warrant evidence persuasiveness/justification claim counterclaim) for the user's cluster

        await self.reflect(topic=current_topic, message_history=message_history)


    async def debate_step(self, websocket: WebSocket, data, app_state):
        payload = json.loads(data)
        message = payload["message"]

        # create a new message id, with 36 characters max
        message_id = str(uuid4())

        # check for collisions
        while self.has_message_id(message_id):
            # re-roll a new message id, with 36 characters max
            message_id = str(uuid4())

        user_id = payload.get("user_id", "")
        session_id = payload.get("session_id", "")

        user_id = app_state.get_value("user_id", "")
        session_id = app_state.get_value("session_id", "")

        # if no user_id, bail
        if user_id == "":
            await websocket.send_json({"status": "error", "response": "Invalid JSON payload"})
            return


        # if no session_id, bail

        if session_id == "":
            await websocket.send_json({"status": "error", "response": "Invalid JSON payload"})
            return

        # default to app state if not provided
        if user_id == "":
            user_id = app_state.get_value("user_id", "")

        message_history = payload["message_history"]
        model = payload.get("model", "solar")
        temperature = float(payload.get("temperature", 0.04))
        current_topic = payload.get("topic", "Unknown")

        prior_ontology = app_state.get_value("prior_ontology", [])

        # if prior_ontology is []:
        #     prior_ontology = []

        current_ontology = self.get_ontology(user_id, session_id, message_id, message)

        mermaid_to_ascii = self.ontological_feature_detection.mermaid_to_ascii(current_ontology)

        print(f"[ prior_ontology: {prior_ontology} ]")

        print(f"[ current_ontology: {current_ontology} ]")

        print(f"[ mermaid_to_ascii: {mermaid_to_ascii} ]")

        prior_ontology.append(current_ontology)

        app_state.set_state("prior_ontology", prior_ontology)

        # algo approach(es):

        # ontological feature detection

        # break previous messages into ontology
        # cache ontology
        # ** use diffuser to spot differentials
        # *** map causal ontology back to spot reference point
        # break current message into ontology
        # ** BLEU score a 10x return on the ontology
        # read ontology + newest

        # await self.think(topic="Chess vs Checkers", prior_ontology=prior_ontology)

        await self.reflect(topic=current_topic, message_history=message_history)


        # topic3:
        # a hat is worn by a person, who has an edge in a meeting due to wearing the hat

        # topic2:
        # I think larger nuclear reactors are better than smaller ones

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

        user_prompt = f"{message}"

        # print(f"\t[ system prompt :: {system_prompt} ]")
        print(f"\t[ user prompt :: {user_prompt} ]")
        simp_msg_history = [{'role': 'system', 'content': system_prompt}]

        # Simplify message history to required format
        for index, message in enumerate(message_history):
            message_role = message['role']
            if message_role == "user":
                message_user_id = f"{message['data']['user_id']}:"
                message_content = message['data']['content']
            else:
                message_user_id = ""
                message_content = message['content']

            simplified_message = {'role': message['role'], 'content': f"{message_user_id}{message_content}"}
            if 'images' in message:
                simplified_message['images'] = message['images']

            simp_msg_history.append(simplified_message)

        simp_msg_history.append({'role': 'user', 'content': f"{user_id}:{user_prompt}"})

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
        semantic_category = self.semantic_compression.fetch_semantic_category(output_combined)

        # Send the final completed message
        await websocket.send_json(
            {"status": "completed", "response": output_combined, "semantic_category": semantic_category.content,
             "completed": True})

    def embed_text(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt')

        # Get the hidden states from the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use the [CLS] token representation as the embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy().squeeze()
        return cls_embedding

    def compute_semantic_distances(self, ontology, topic_embedding):
        # Embed the ontology text
        ontology_embedding = self.embed_text(ontology)

        # Compute cosine similarity between ontology and topic
        similarity = cosine_similarity([ontology_embedding], [topic_embedding])[0][0]

        return 1 - similarity  # Return distance instead of similarity

    def normalize_distances(self, distances):
        total = np.sum(distances)
        if total == 0:
            return np.zeros_like(distances)
        return distances / total

    def aggregate_distributions(self, semantic_distances):
        # Convert to numpy array for easier manipulation
        distances = np.array(semantic_distances)

        # Compute the mean across all distances to form the collective distribution
        if len(distances) == 0:
            return np.array([0.5])  # Handle empty distributions
        collective_distribution = np.mean(distances, axis=0)

        return collective_distribution

    def calculate_kl_divergence(self, p, q):
        # Ensure the distributions are numpy arrays
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        # Normalize the distributions
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Calculate the KL-Divergence
        kl_div = entropy(p, q)

        return kl_div

    def calculate_impact_scores(self, kl_divergences, collective_distribution):
        # Calculate the impact score for each contribution
        impact_scores = []
        for kl_divergence in kl_divergences:
            impact_score = kl_divergence - collective_distribution
            impact_scores.append(impact_score)

        return impact_scores

    async def think(self, topic, prior_ontology):
        print(f"\t[ think :: topic :: {topic} ]")
        print(f"\t[ think :: prior_ontology :: {prior_ontology} ]")

        # Embed the topic
        topic_embedding = self.embed_text(topic)

        # Compute semantic distances for each contribution
        semantic_distances = []
        for ontology in prior_ontology:
            distance = self.compute_semantic_distances(ontology, topic_embedding)
            print(f"\t\t[ think :: distance :: {distance} ]")
            semantic_distances.append(distance)

        # Normalize distances
        normalized_distances = self.normalize_distances(semantic_distances)
        print(f"\t[ think :: normalized_distances :: {normalized_distances} ]")

        # Aggregate the distributions to form a collective distribution
        collective_distribution = self.aggregate_distributions(normalized_distances)
        print(f"\t[ think :: collective_distribution :: {collective_distribution} ]")

        # Calculate KL-Divergence for each contribution
        kl_divergences = []
        for distance in normalized_distances:
            kl_divergence = self.calculate_kl_divergence([distance, 1 - distance],
                                                         [collective_distribution, 1 - collective_distribution])
            print(f"\t\t[ think :: kl_divergence :: {kl_divergence} ]")
            kl_divergences.append(kl_divergence)

        # Calculate impact scores
        impact_scores = self.calculate_impact_scores(kl_divergences, collective_distribution)
        print(f"\t[ think :: impact_scores :: {impact_scores} ]")

        # Store results in app_state (subkey session_id)
        app_state = AppState().get_instance()
        app_state.set_state("kl_divergences", kl_divergences)

        print(f"\t[ think :: kl_divergences :: {kl_divergences} ]")

        parsed_ontology = [self.parse_mermaid_to_dict(component) for component in prior_ontology]
        print(f"\t[ think :: parsed_ontology :: {parsed_ontology} ]")

        # Build a graph from parsed ontology
        graph = self.build_graph(parsed_ontology)
        print(f"\t[ think :: graph :: {graph} ]")

        # Calculate weights for each vertex based on the number of edges
        vertex_weights = self.calculate_vertex_weights(graph)
        print(f"\t[ think :: vertex_weights :: {vertex_weights} ]")

        # Identify and weigh densely connected sub-graphs
        sub_graph_weights = self.calculate_sub_graph_weights(graph)
        print(f"\t[ think :: sub_graph_weights :: {sub_graph_weights} ]")

        # Combine weights to determine the strength of each argument
        argument_weights = self.combine_weights(vertex_weights, sub_graph_weights)
        print(f"\t[ think :: argument_weights :: {argument_weights} ]")

        # Rank arguments based on their weights
        ranked_arguments = self.rank_arguments(argument_weights)
        print(f"\t[ think :: ranked_arguments :: {ranked_arguments} ]")

        return ranked_arguments

    def parse_mermaid_to_dict(self, mermaid_str):
        """Parse mermaid flowchart syntax into a dictionary with 'relations'."""
        lines = mermaid_str.strip().split("\n")
        relations = []
        for line in lines:
            if "-->" in line:
                parts = line.split("-->")
                subject = parts[0].strip().split('[')[0]
                relation_type = "less complicated" if "|\"less complicated\"|" in parts[1] else "relation"
                obj = parts[1].strip().split('[')[0].split('|')[0].strip()
                relations.append((subject, relation_type, obj))
        return {"relations": relations}

    def build_graph(self, ontology):
        print(f"\t[ build_graph :: start ]")
        graph = {}
        for index, component in enumerate(ontology):
            print(f"\t[ build_graph :: component[{index}] :: {component} ]")
            if isinstance(component, dict) and 'relations' in component:
                for relation in component['relations']:
                    subject, relation_type, obj = relation
                    print(f"\t[ build_graph :: relation :: {subject} --{relation_type}--> {obj} ]")
                    if subject not in graph:
                        graph[subject] = []
                    if obj not in graph:
                        graph[obj] = []
                    graph[subject].append(obj)
                    graph[obj].append(subject)
            else:
                print(f"\t[ build_graph :: error :: component[{index}] is not a dict or missing 'relations' key ]")
        print(f"\t[ build_graph :: graph :: {graph} ]")
        return graph

    def calculate_vertex_weights(self, graph):
        print(f"\t[ calculate_vertex_weights :: start ]")
        vertex_weights = {vertex: len(edges) for vertex, edges in graph.items()}
        for vertex, weight in vertex_weights.items():
            print(f"\t[ calculate_vertex_weights :: vertex :: {vertex} :: weight :: {weight} ]")
        return vertex_weights

    def calculate_sub_graph_weights(self, graph):
        print(f"\t[ calculate_sub_graph_weights :: start ]")
        sub_graph_weights = {}
        visited = set()
        for vertex in graph:
            if vertex not in visited:
                sub_graph = self.dfs(graph, vertex, visited)
                sub_graph_weight = sum(self.calculate_vertex_weights(sub_graph).values())
                sub_graph_weights[vertex] = sub_graph_weight
                print(f"\t[ calculate_sub_graph_weights :: vertex :: {vertex} :: sub_graph_weight :: {sub_graph_weight} ]")
        return sub_graph_weights

    def dfs(self, graph, start, visited):
        print(f"\t[ dfs :: start :: {start} ]")
        stack = [start]
        sub_graph = {}
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                sub_graph[vertex] = graph[vertex]
                stack.extend([v for v in graph[vertex] if v not in visited])
                print(f"\t[ dfs :: vertex :: {vertex} :: visited :: {visited} ]")
        print(f"\t[ dfs :: sub_graph :: {sub_graph} ]")
        return sub_graph

    def combine_weights(self, vertex_weights, sub_graph_weights):
        print(f"\t[ combine_weights :: start ]")
        combined_weights = {}
        for vertex in vertex_weights:
            combined_weights[vertex] = vertex_weights[vertex] + sub_graph_weights.get(vertex, 0)
            print(f"\t[ combine_weights :: vertex :: {vertex} :: combined_weight :: {combined_weights[vertex]} ]")
        return combined_weights

    def rank_arguments(self, argument_weights):
        print(f"\t[ rank_arguments :: start ]")
        ranked_arguments = sorted(argument_weights.items(), key=lambda item: item[1], reverse=True)
        for rank, (vertex, weight) in enumerate(ranked_arguments, 1):
            print(f"\t[ rank_arguments :: rank :: {rank} :: vertex :: {vertex} :: weight :: {weight} ]")
        return ranked_arguments

    def cluster_messages(self, user_messages):
        clustered_messages = {}
        for user_id, messages in user_messages.items():
            if len(messages) > 1:
                clusters = self.argument_detection.cluster_sentences(messages, distance_threshold=1.45)
                clustered_messages[user_id] = clusters
        return clustered_messages

    def wepcc_cluster(self, clustered_messages):
        wepcc_results = {}
        for user_id, clusters in clustered_messages.items():
            wepcc_results[user_id] = {}
            for cluster_id, cluster_sentences in clusters.items():
                print(f"\t[ reflect :: Running WEPCC for user {user_id}, cluster {cluster_id} ]")
                warrant, evidence, persuasiveness_justification, claim, counterclaim = self.argument_detection.fetch_argument_definition(
                    cluster_sentences)
                wepcc_results[user_id][cluster_id] = {
                    'warrant': warrant,
                    'evidence': evidence,
                    'persuasiveness_justification': persuasiveness_justification,
                    'claim': claim,
                    'counterclaim': counterclaim
                }
                print(
                    f"\t[ reflect :: WEPCC for user {user_id}, cluster {cluster_id} :: {wepcc_results[user_id][cluster_id]} ]")
        return wepcc_results

    def get_cluster_weight_modulator(self, wepcc_results, cutoff):
        cluster_weight_modulator = {}
        for user_idA, clustersA in wepcc_results.items():
            cluster_weight_modulator[user_idA] = cluster_weight_modulator.get(user_idA, {})

            for cluster_idA, wepccA in clustersA.items():
                phase_sim_A = []
                for user_idB, clustersB in wepcc_results.items():
                    if user_idA != user_idB:
                        for cluster_idB, wepccB in clustersB.items():
                            # Calculate cosine similarity between counterclaims and claims
                            counterclaim_embedding = self.fast_embedding_model.encode(wepccA['counterclaim'])
                            claim_embedding = self.fast_embedding_model.encode(wepccB['claim'])
                            sim_score = cosine_similarity([counterclaim_embedding], [claim_embedding])[0][0]
                            print(
                                f"\t[ reflect :: Sim score between {user_idA}'s counterclaim (cluster {cluster_idA}) and {user_idB}'s claim (cluster {cluster_idB}) :: {sim_score} ]")
                            if sim_score > cutoff:
                                phase_sim_A.append((sim_score, cluster_idB, user_idB))
                if cluster_idA not in cluster_weight_modulator[user_idA]:
                    cluster_weight_modulator[user_idA][cluster_idA] = []
                for sim_score, cluster_idB, user_idB in phase_sim_A:
                    normalized_value = (sim_score - cutoff) / (1 - cutoff)
                    cluster_weight_modulator[user_idA][cluster_idA].append(normalized_value)
                    print(
                        f"\t[ reflect :: Normalized value for {user_idA} (cluster {cluster_idA}) :: {normalized_value} ]")
        return cluster_weight_modulator

    def gather_final_results(self, cluster_shadow_coverage, wepcc_results, unaddressed_score_multiplier):
        aggregated_scores = {}
        addressed_clusters = {}
        unaddressed_clusters = {}

        results = []

        for user_id, weight_mods in cluster_shadow_coverage.items():
            total_score = 0
            addressed_clusters[user_id] = []
            unaddressed_clusters[user_id] = []

            user_result = {"user": user_id, "clusters": []}

            for cluster_id, modulator in weight_mods.items():
                try:
                    persuasiveness_object = json.loads(
                        wepcc_results[user_id][cluster_id]['persuasiveness_justification'])
                    persuasiveness_score = float(persuasiveness_object['content']['persuasiveness_score'])
                    addressed_score = (1 - modulator) * persuasiveness_score
                    total_score += addressed_score
                    addressed_clusters[user_id].append((cluster_id, addressed_score))
                    user_result["clusters"].append({
                        "cluster": cluster_id,
                        "type": "addressed",
                        "score": addressed_score
                    })
                    print(
                        f"\t[ reflect :: Addressed score for User {user_id}, Cluster {cluster_id} :: {addressed_score} ]")
                except json.JSONDecodeError as e:
                    print(f"\t[ reflect :: JSONDecodeError for User {user_id}, Cluster {cluster_id} :: {e} ]")
                    print(
                        f"\t[ reflect :: Invalid JSON :: {wepcc_results[user_id][cluster_id]['persuasiveness_justification']} ]")

            # Add unaddressed arguments' scores
            for cluster_id, wepcc in wepcc_results[user_id].items():
                if cluster_id not in weight_mods:
                    try:
                        persuasiveness_object = json.loads(wepcc['persuasiveness_justification'])
                        persuasiveness_score = float(persuasiveness_object['content']['persuasiveness_score'])
                        unaddressed_score = persuasiveness_score * unaddressed_score_multiplier
                        total_score += unaddressed_score
                        unaddressed_clusters[user_id].append((cluster_id, unaddressed_score))
                        user_result["clusters"].append({
                            "cluster": cluster_id,
                            "type": "unaddressed",
                            "score": unaddressed_score
                        })
                        print(
                            f"\t[ reflect :: Unaddressed score for User {user_id}, Cluster {cluster_id} :: {unaddressed_score} ]")
                    except json.JSONDecodeError as e:
                        print(f"\t[ reflect :: JSONDecodeError for User {user_id}, Cluster {cluster_id} :: {e} ]")
                        print(f"\t[ reflect :: Invalid JSON :: {wepcc['persuasiveness_justification']} ]")

            aggregated_scores[user_id] = total_score
            user_result["total_score"] = total_score
            results.append(user_result)
            print(f"\t[ reflect :: Aggregated score for User {user_id} :: {total_score} ]")

        return aggregated_scores, addressed_clusters, unaddressed_clusters, results

    def get_cluster_shadow_coverage(self, cluster_weight_modulator, cutoff):
        final_scores = {}

        # Post-process the collected normalized values for each cluster
        for user_id, cluster_data in cluster_weight_modulator.items():
            final_scores[user_id] = final_scores.get(user_id, {})
            for cluster_idA, normalized_values in cluster_data.items():
                if normalized_values:
                    highest = max(normalized_values)
                    shadow_coverage = highest
                    for value in normalized_values:
                        if value != highest:
                            shadow_coverage += (value * (1.0 - cutoff)) * (1 - shadow_coverage)
                            # Since we're adding coverage, shadow_coverage should naturally stay within [0,1]
                            # No need to clamp or use min

                    # Initialize the nested dictionary if it doesn't exist
                    if cluster_idA not in final_scores[user_id]:
                        final_scores[user_id][cluster_idA] = 0

                    # Store the final score
                    final_scores[user_id][cluster_idA] = shadow_coverage
                    print(
                        f"\t[ reflect :: Combined score for {user_id} (cluster {cluster_idA}) :: {shadow_coverage} ]")

        return final_scores

    async def reflect(self, topic, message_history):
        unaddressed_score_multiplier = 2.5

        print(f"\t[ reflect :: topic :: {topic} ]")

        # Check if there are at least two users with at least one cluster each
        if len(clustered_messages) < 2 or any(len(clusters) < 1 for clusters in clustered_messages.values()):
            print("\t[ reflect :: Not enough clusters or users to perform argument matching ]")
            return

        # Step 3: Run WEPCC on each cluster
        wepcc_results = self.wepcc_cluster(clustered_messages)
        print(f"\t[ reflect :: wepcc_results :: {wepcc_results} ]")

        # Define similarity cutoff threshold
        cutoff = 0.5

        # Initialize phase similarity and cluster weight modulator
        # Step 4: Match each user's Counterclaims with all other users' Claims
        cluster_weight_modulator = self.get_cluster_weight_modulator(wepcc_results, cutoff)

        # Step 5: Calculate the counter-factual shadow coverage for each cluster
        # Create a new dictionary to hold the final combined scores
        cluster_shadow_coverage = self.get_cluster_shadow_coverage(cluster_weight_modulator, cutoff)

        # Step 6: Final aggregation and ranking
        # Final aggregation and ranking
        (aggregated_scores,
         addressed_clusters,
         unaddressed_clusters,
         results) = self.gather_final_results(cluster_shadow_coverage, wepcc_results, unaddressed_score_multiplier)

        print(f"\t[ reflect :: aggregated_scores :: {aggregated_scores} ]")
        print(f"\t[ reflect :: addressed_clusters :: {addressed_clusters} ]")
        print(f"\t[ reflect :: unaddressed_clusters :: {unaddressed_clusters} ]")

        app_state = AppState().get_instance()
        app_state.set_state("wepcc_results", wepcc_results)
        app_state.set_state("aggregated_scores", aggregated_scores)
        app_state.set_state("addressed_clusters", addressed_clusters)
        app_state.set_state("unaddressed_clusters", unaddressed_clusters)

        print(f"\t[ reflect :: Completed ]")

        return results

#alright! once again, same style, same acumen, boil over each and every one of those

#okay compose it all in a series of functions so I can copy paste.

#AFTERWARDS I'd like a list of all of the new functions you need to yet provide super-stubs for