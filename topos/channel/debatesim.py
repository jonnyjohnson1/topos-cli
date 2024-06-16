# topos/channel/debatesim.py

import os
from uuid import uuid4

from dotenv import load_dotenv

import json
from datetime import datetime
import time

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import entropy

from fastapi import WebSocket, WebSocketDisconnect
from topos.FC.semantic_compression import SemanticCompression
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
    def __init__(self):
        # Load the pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')

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

        user_id = payload.get("user_id", "")
        session_id = payload.get("session_id", "")

        # default to app state if not provided
        if user_id == "":
            user_id = app_state.get_value("user_id", "")
        if session_id == "":
            session_id = app_state.get_value("session_id", "")

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

        await self.reflect(topic=current_topic, message_history=message_history, llm_model="ollama:dolphin-llama3")


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
        semantic_compression = SemanticCompression(model=f"ollama:{model}", api_key="ollama")
        semantic_category = semantic_compression.fetch_semantic_category(output_combined)

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

    async def reflect(self, topic, message_history, llm_model):
        print(f"\t[ reflect :: topic :: {topic} ]")

        argument_detection = ArgumentDetection("ollama", llm_model)

        # Step 1: Gather message history for specific users
        user_messages = {}
        for message in message_history:
            user_id = message['data']['user_id']
            content = message['data']['content']
            if user_id not in user_messages:
                user_messages[user_id] = []
            user_messages[user_id].append(content)

        print(f"\t[ reflect :: user_messages :: {user_messages} ]")

        should_continue = False

        # Step 2: Cluster analysis for each user's messages
        clustered_messages = {}
        for user_id, messages in user_messages.items():
            if (len(messages) > 1):
                print(f"\t[ reflect :: Clustering messages for user {user_id} ]")
                clusters = argument_detection.cluster_sentences(messages, distance_threshold=1.45)
                clustered_messages[user_id] = clusters

        print(f"\t[ reflect :: clustered_messages :: {clustered_messages} ]")

        if len(clustered_messages.keys()) > 0:
            # Step 3: Run WEPCC on each cluster
            wepcc_results = {}
            for user_id, clusters in clustered_messages.items():
                wepcc_results[user_id] = {}
                for cluster_id, cluster_sentences in clusters.items():
                    print(f"\t[ reflect :: Running WEPCC for user {user_id}, cluster {cluster_id} ]")
                    warrant, evidence, persuasiveness_justification, claim, counterclaim = argument_detection.fetch_argument_definition(
                        cluster_sentences)
                    wepcc_results[user_id][cluster_id] = {
                        'warrant': warrant,
                        'evidence': evidence,
                        'persuasiveness_justification': persuasiveness_justification,
                        'claim': claim,
                        'counterclaim': counterclaim
                    }

                print(f"\t[ reflect :: wepcc_results :: {wepcc_results} ]")

                # Here you would typically update the app state or further process the WEPCC results
                app_state = AppState().get_instance()
                app_state.set_state("wepcc_results", wepcc_results)

        # You can also add logic to evaluate the debate, score the arguments, and determine the winning side based on the WEPCC results.

        print(f"\t[ reflect :: Completed ]")

#alright! once again, same style, same acumen, boil over each and every one of those

#okay compose it all in a series of functions so I can copy paste.

#AFTERWARDS I'd like a list of all of the new functions you need to yet provide super-stubs for