# topos/channel/debatesim.py
import hashlib
import asyncio

import traceback

from typing import Dict, List
import pprint

import os
from queue import Queue

from datetime import datetime, timedelta, UTC
import time

from dotenv import load_dotenv

from uuid import uuid4

import json
import jwt
from jwt.exceptions import InvalidTokenError

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np
from scipy.stats import entropy

from fastapi import WebSocket, WebSocketDisconnect
from ..FC.argument_detection import ArgumentDetection
from ..config import get_openai_api_key
from ..models.llm_classes import vision_models
from ..services.database.app_state import AppState
from ..utilities.utils import create_conversation_string
from ..services.classification_service.base_analysis import base_text_classifier, base_token_classifier
from topos.FC.conversation_cache_manager import ConversationCacheManager
from topos.FC.semantic_compression import SemanticCompression
from topos.FC.ontological_feature_detection import OntologicalFeatureDetection

from topos.channel.channel_engine import ChannelEngine

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


class Cluster:
    def __init__(self, cluster_id, sentences, user_id, generation, session_id, coherence):
        self.cluster_id = cluster_id
        self.sentences = sentences
        self.cluster_hash = self.generate_hash()
        self.user_id = user_id
        self.generation = generation
        self.session_id = session_id
        self.coherence = coherence
        self.wepcc_result = None

    def generate_hash(self):
        sorted_sentences = sorted(self.sentences)
        return hashlib.sha256(json.dumps(sorted_sentences).encode()).hexdigest()

    def to_dict(self):
        return {
            "cluster_id": self.cluster_id,
            "sentences": self.sentences,
            "cluster_hash": self.cluster_hash,
            "user_id": self.user_id,
            "generation": self.generation,
            "session_id": self.session_id,
            "coherence": self.coherence,
            "wepcc_result": self.wepcc_result,
        }

    def update_wepcc(self, wepcc_result):
        self.wepcc_result = wepcc_result



class DebateSimulator:
    _instance = None
    _lock = asyncio.Lock()

    @staticmethod
    async def get_instance():
        if DebateSimulator._instance is None:
            async with DebateSimulator._lock:
                if DebateSimulator._instance is None:
                    DebateSimulator._instance = DebateSimulator()
        return DebateSimulator._instance

    def __init__(self, use_neo4j=False):
        if DebateSimulator._instance is not None:
            raise Exception("This class is a singleton!")
        else:

            if AppState._instance is None:
                AppState(use_neo4j=use_neo4j)

            load_dotenv()  # Load environment variables

            # Load the pre-trained model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')

            self.operational_llm_model = "ollama:dolphin-llama3"
            self.argument_detection_llm_model = "ollama:dolphin-llama3"
            # self.argument_detection_llm_model = "claude:claude-3-5-sonnet-20240620"
            # self.argument_detection_llm_model = "openai:gpt-4o"

            ONE_API_API_KEY = os.getenv("ONE_API_API_KEY")

            # Initialize the SentenceTransformer model for embedding text
            # self.fast_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.fast_embedding_model = SentenceTransformer('all-mpnet-base-v2')

            self.argument_detection = ArgumentDetection(model=self.argument_detection_llm_model, api_key=ONE_API_API_KEY)

            self.semantic_compression = SemanticCompression(model=self.operational_llm_model, api_key="ollama")

            self.app_state = AppState.get_instance()

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

            self.current_generation = None
            self.websocket_groups = {}

            self.channel_engine = ChannelEngine()
            self.channel_engine.register_task_handler('check_and_reflect', self.check_and_reflect)  # Register the handler
            self.channel_engine.register_task_handler('broadcast', self.websocket_broadcast)

    def generate_jwt_token(self, user_id, session_id):
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "exp": datetime.now(UTC) + timedelta(hours=1)  # Token valid for 1 hour
        }
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        return token

    async def add_to_websocket_group(self, session_id, websocket):
        await self._lock.acquire()
        try:
            if session_id not in self.websocket_groups:
                self.websocket_groups[session_id] = []
            self.websocket_groups[session_id].append(websocket)
        finally:
            self._lock.release()

    async def remove_from_websocket_group(self, session_id, websocket):
        await self._lock.acquire()
        try:
            if session_id in self.websocket_groups:
                self.websocket_groups[session_id].remove(websocket)
                if not self.websocket_groups[session_id]:
                    del self.websocket_groups[session_id]
        finally:
            self._lock.release()

    # async def execute_task(self, task):
    #     print(f"Executing task: {task['type']}")
    #     if task['type'] == 'check_and_reflect':
    #         self.current_generation = task['generation_nonce']
    #         await self.check_and_reflect(task['session_id'], task['user_id'], task['generation_nonce'],
    #                                      task['message_id'], task['message'])
    #     elif task['type'] == 'broadcast':
    #         await self.websocket_broadcast(task['websocket'], task['message'])
    #     # print(f"Finished executing task: {task['type']}")

    async def websocket_broadcast(self, websocket, message):
        if message:
            await websocket.send_text(message)

    async def stop_all_reflect_tasks(self):
        await self.channel_engine.reset_processing_queue()

    async def get_ontology(self, user_id, session_id, message_id, message):
        composable_string = f"for user {user_id}, of {session_id}, the message is: {message}"
        # print(f"\t\t[ composable_string :: {composable_string} ]")

        entities, pos_tags, dependencies, relations, srl_results, timestamp, context_entities = self.ontological_feature_detection.build_ontology_from_paragraph(
            user_id, session_id, message_id, composable_string)

        if self.use_neo4j:
            self.ontological_feature_detection.store_ontology(user_id, session_id, message_id, message, timestamp, context_entities, relations)

        input_components = message, entities, dependencies, relations, srl_results, timestamp, context_entities

        mermaid_syntax = self.ontological_feature_detection.extract_mermaid_syntax(input_components, input_type="components")
        return mermaid_syntax

    def has_message_id(self, message_id):
        if self.use_neo4j:
            return self.ontological_feature_detection.check_message_exists(message_id)
        else:
            return False

    async def integrate(self, token, data, app_state, cancel_old_tasks):
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
        except InvalidTokenError:
            print(f"Invalid JWT token error :: {token}")
            # await websocket.send_json({"status": "error", "response": "Invalid JWT token"})
            return

        session_id = payload.get("session_id", "")

        # if no user_id, bail
        if user_id == "" or session_id == "":
            return

        current_topic = payload.get("topic", "Unknown")

        # from app state
        message_history = app_state.get_value(f"message_history_{session_id}", [])

        prior_ontology = app_state.get_value(f"prior_ontology_{session_id}", [])

        current_ontology = await self.get_ontology(user_id, session_id, message_id, message)

        # print(f"[ prior_ontology: {prior_ontology} ]")
        # print(f"[ current_ontology: {current_ontology} ]")

        prior_ontology.append(current_ontology)

        app_state.set_state(f"prior_ontology_{session_id}", prior_ontology)

        mermaid_to_ascii = self.ontological_feature_detection.mermaid_to_ascii(current_ontology)
        # print(f"[ mermaid_to_ascii: {mermaid_to_ascii} ]")

        new_history_item = {
            "data": {
                "user_id": user_id,
                "content": message,
                "timestamp": datetime.now().isoformat(),
                "message_id": message_id,
                "topic": current_topic
            },
            "ontology": current_ontology,
            "mermaid": mermaid_to_ascii
        }

        message_history.append(new_history_item)

        app_state.set_state(f"message_history_{session_id}", message_history)

        # Create new Generation
        generation_nonce = self.generate_nonce()

        if cancel_old_tasks:
            await self.stop_all_reflect_tasks()

        # print(f"Creating check_and_reflect task for message: {message_id}")
        task = {
            'type': 'check_and_reflect',
            'session_id': session_id,
            'user_id': user_id,
            'generation_nonce': generation_nonce,
            'message_id': message_id,
            'message': message
        }
        print(f"Task created: {task}")
        await self.channel_engine.add_task(task)
        # print(f"Task added to queue for message: {message_id}")

        return current_ontology, message_id

    @staticmethod
    def generate_nonce():
        return str(uuid4())

    @staticmethod
    def break_into_sentences(messages, min_words=20):
        output = []
        for message in messages:
            content = message["data"]["content"].strip()  # Remove leading/trailing whitespace
            sentences = sent_tokenize(content)

            current_sentence = []

            for sentence in sentences:
                sentence = sentence.strip()  # Remove leading/trailing whitespace
                if not sentence:
                    continue  # Skip empty sentences

                words = sentence.split()
                if len(current_sentence) + len(words) >= min_words:
                    current_sentence.extend(words)  # Extend current sentence before appending
                    output.append({"role": message["role"], "data": {"user_id": message["data"]["user_id"],
                                                                     "content": " ".join(current_sentence)}})
                    current_sentence = []  # Reset current_sentence after appending
                else:
                    current_sentence.extend(words)

            if current_sentence:
                output.append({"role": message["role"],
                               "data": {"user_id": message["data"]["user_id"], "content": " ".join(current_sentence)}})

        return output

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

    def incremental_clustering(self, clusters, previous_clusters):
        updated_clusters = {}
        for user_id, user_clusters in clusters.items():
            if user_id not in previous_clusters:
                updated_clusters[user_id] = user_clusters
            else:
                updated_clusters[user_id] = {}
                for cluster_id, cluster in user_clusters.items():
                    previous_cluster_hash = previous_clusters[user_id].get(cluster_id, None)
                    if not previous_cluster_hash or cluster.cluster_hash != previous_cluster_hash.cluster_hash:
                        updated_clusters[user_id][cluster_id] = cluster

        return updated_clusters

    async def broadcast_to_websocket_group(self, session_id, json_message):
        await self._lock.acquire()
        try:
            websockets = self.websocket_groups.get(session_id, [])

            for websocket in websockets:
                await websocket.send_json(json_message)
        finally:
            self._lock.release()

    def check_generation_halting(self, generation_nonce):
        if self.current_generation is not None and self.current_generation != generation_nonce:
            return True

        return False

    async def check_and_reflect(self, session_id, user_id, generation_nonce, message_id, message):
        print(f"\t[ check_and_reflect started for message: {message_id} ]")
        # "Reflect"
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

        app_state = AppState().get_instance()

        message_history = app_state.get_value(f"message_history_{session_id}", [])

        # Step 1: Gather message history for specific users
        user_messages = self.aggregate_user_messages(message_history)
        # print(f"\t[ reflect :: user_messages :: {user_messages} ]")

        # Step 2: Cluster analysis for each user's messages
        clusters = self.cluster_messages(user_messages, generation_nonce, session_id)
        # print(f"\t[ reflect :: clustered_messages :: {len(clusters)} ]")

        # Send initial cluster data back to frontend
        await self.broadcast_to_websocket_group(session_id, {
            "status": "initial_clusters",
            "clusters": {user_id: [cluster.to_dict() for cluster in user_clusters.values()] for user_id, user_clusters
                         in clusters.items()},
            "generation": generation_nonce
        })
        if self.check_generation_halting(generation_nonce) is True:
            return

        # Perform incremental clustering if needed
        previous_clusters = app_state.get_value(f"previous_clusters_{session_id}_{user_id}", {})

        # Extract properly ID-matching clusters from previous_clusters
        for user_id, user_clusters in clusters.items():
            if user_id in previous_clusters:
                for cluster_id, cluster in user_clusters.items():
                    if cluster_id in previous_clusters[user_id]:
                        cluster.update_wepcc(previous_clusters[user_id][cluster_id].wepcc_result)

        updated_clusters = self.incremental_clustering(clusters, previous_clusters)

        # Send updated cluster data back to frontend
        await self.broadcast_to_websocket_group(session_id, {
            "status": "updated_clusters",
            "clusters": {user_id: [cluster.to_dict() for cluster in user_clusters.values()] for user_id, user_clusters
                         in updated_clusters.items()},
            "generation": generation_nonce
        })
        if self.check_generation_halting(generation_nonce) is True:
            return

        async def report_wepcc_result(generation_nonce, user_id, cluster_id, cluster_hash, wepcc_result):
            await self.broadcast_to_websocket_group(session_id, {
                "status": "wepcc_result",
                "generation": generation_nonce,
                "user_id": user_id,
                "cluster_id": cluster_id,
                "cluster_hash": cluster_hash,
                "wepcc_result": wepcc_result,
            })
            if self.check_generation_halting(generation_nonce) is True:
                return

        # Step 3: Run WEPCC on each cluster
        # these each take a bit to process, so we're passing in the websocket group to stream the results back out
        # due to timing these may be inconsequential re: generation, but they're going to send back the results anyhow.
        wepcc_results = await self.wepcc_cluster(updated_clusters, report_wepcc_result)
        # print(f"\t[ reflect :: wepcc_results :: {wepcc_results} ]")

        # Update clusters with WEPCC results
        for user_id, user_clusters in updated_clusters.items():
            for cluster_id, cluster in user_clusters.items():
                if cluster_id in wepcc_results[user_id]:
                    wepcc = wepcc_results[user_id][cluster_id]
                    cluster.update_wepcc(wepcc)

        app_state.set_state(f"previous_clusters_{session_id}_{user_id}", clusters)

        # Check if there are enough clusters or users to perform argument matching
        if len(clusters) < 2:
            print("\t[ reflect :: Not enough clusters, but returning user's clusters ]")

            # Initialize shadow coverage with no coverage
            cluster_shadow_coverage = {user_id: {} for user_id in clusters.keys()}

            # Assume wepcc_results are already calculated earlier in the process
            unaddressed_score_multiplier = 2.5  # Example multiplier

            # Call gather_final_results
            aggregated_scores, addressed_clusters, unaddressed_clusters, results = self.gather_final_results(
                cluster_shadow_coverage, clusters, unaddressed_score_multiplier
            )

            await self.broadcast_to_websocket_group(session_id, {
                "status": "final_results",
                "generation": generation_nonce,
                "aggregated_scores": aggregated_scores,
                "addressed_clusters": addressed_clusters,
                "unaddressed_clusters": unaddressed_clusters,
                "results": results,
            })

            return

        # Define similarity cutoff threshold
        cutoff = 0.35

        # Define unaddressed score multiplier
        unaddressed_score_multiplier = 2.5

        # Initialize phase similarity and cluster weight modulator
        # Step 4: Match each user's Counterclaims with all other users' Claims
        # This function takes a moment, as it does an embedding check. Not super heavy, but with enough participants
        # certainly an async operation
        cluster_weight_modulator = self.get_cluster_weight_modulator(clusters, cutoff)

        # Step 5: Calculate the counter-factual shadow coverage for each cluster
        # Create a new dictionary to hold the final combined scores
        # This function is very fast, relatively speaking
        cluster_shadow_coverage = self.get_cluster_shadow_coverage(cluster_weight_modulator, cutoff)

        # Step 6: Final aggregation and ranking
        (aggregated_scores,
         addressed_clusters,
         unaddressed_clusters,
         results) = self.gather_final_results(cluster_shadow_coverage, clusters, unaddressed_score_multiplier)

        print(f"\t[ reflect :: aggregated_scores :: {aggregated_scores} ]")
        print(f"\t[ reflect :: addressed_clusters :: {addressed_clusters} ]")
        print(f"\t[ reflect :: unaddressed_clusters :: {unaddressed_clusters} ]")

        # Print the number of unaddressed clusters for each user
        print("\nUnaddressed Clusters Summary:")
        for user_id, unaddressed_list in unaddressed_clusters.items():
            num_unaddressed = len(unaddressed_list)
            print(f"\t\t[ User {user_id}: {num_unaddressed} unaddressed cluster(s) ]")


        app_state.set_state("wepcc_results", wepcc_results)
        app_state.set_state("aggregated_scores", aggregated_scores)
        app_state.set_state("addressed_clusters", addressed_clusters)
        app_state.set_state("unaddressed_clusters", unaddressed_clusters)

        await self.broadcast_to_websocket_group(session_id, {
            "status": "final_results",
            "generation": generation_nonce,
            "aggregated_scores": aggregated_scores,
            "addressed_clusters": addressed_clusters,
            "unaddressed_clusters": unaddressed_clusters,
            "results": results,
        })

        print(f"\t[ check_and_reflect :: Completed ]")

    def cluster_messages(self, user_messages, generation, session_id):
        clustered_messages = {}
        for user_id, messages in user_messages.items():
            if len(messages) > 1:
                clusters, coherence_scores = self.argument_detection.cluster_sentences(messages, distance_threshold=0.3)
                clustered_messages[user_id] = {
                    int(cluster_id): Cluster(cluster_id=int(cluster_id),
                                             sentences=cluster_sentences,
                                             user_id=user_id,
                                             generation=generation,
                                             session_id=session_id,
                                             coherence=coherence_scores.get(cluster_id, 1.0))
                    for cluster_id, cluster_sentences in clusters.items()
                }
            elif len(messages) == 1:
                # Create a single cluster for the lone message
                cluster_id = 0
                clustered_messages[user_id] = {
                    cluster_id: Cluster(cluster_id=cluster_id,
                                        sentences=messages,
                                        user_id=user_id,
                                        generation=generation,

                                        session_id=session_id,
                                        coherence=1.0)  # Single message cluster always has perfect coherence
                }
        return clustered_messages

    async def wepcc_cluster(self, clusters: Dict[str, Cluster], report_wepcc_result):
        wepcc_results = {}
        for user_id, user_clusters in clusters.items():
            wepcc_results[user_id] = {}
            for cluster_id, cluster in user_clusters.items():
                # print(f"\t[ reflect :: Running WEPCC for user {user_id}, cluster {cluster_id} ]")
                warrant, evidence, persuasiveness_justification, claim, counterclaim = self.argument_detection.fetch_argument_definition(
                    cluster.sentences)
                wepcc_results[user_id][cluster_id] = {
                    'warrant': warrant,
                    'evidence': evidence,
                    'persuasiveness_justification': persuasiveness_justification,
                    'claim': claim,
                    'counterclaim': counterclaim
                }
                self.pretty_print_wepcc_result(user_id, cluster_id, wepcc_results[user_id][cluster_id])
                # print(
                #     f"\t[ reflect :: WEPCC for user {user_id}, cluster {cluster_id} :: {wepcc_results[user_id][cluster_id]} ]")

                # Output to websocket
                await report_wepcc_result(cluster.cluster_hash, user_id, cluster_id, cluster.cluster_hash,
                                          wepcc_results[user_id][cluster_id])
        return wepcc_results

    def get_cluster_weight_modulator(self, clusters, cutoff):
        cluster_weight_modulator = {}
        for user_idA, user_clustersA in clusters.items():
            cluster_weight_modulator[user_idA] = cluster_weight_modulator.get(user_idA, {})

            for cluster_idA, clusterA in user_clustersA.items():
                phase_sim_A = []
                wepcc_cluster_a = clusterA.wepcc_result
                counterclaim_a = json.loads(wepcc_cluster_a['counterclaim'])
                counterclaim_embedding = self.fast_embedding_model.encode(counterclaim_a['content'])

                for user_idB, user_clustersB in clusters.items():
                    if user_idA != user_idB:
                        for cluster_idB, clusterB in user_clustersB.items():
                            wepcc_cluster_b = clusterB.wepcc_result
                            claim_b = json.loads(wepcc_cluster_b['claim'])

                            # Calculate cosine similarity between counterclaims and claims
                            claim_embedding = self.fast_embedding_model.encode(claim_b['content'])
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

    def gather_final_results(self, cluster_shadow_coverage, clusters, unaddressed_score_multiplier):
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
                    cluster = clusters[user_id][cluster_id]
                    persuasiveness_score = float(
                        json.loads(cluster.wepcc_result['persuasiveness_justification'])['content'][
                            'persuasiveness_score'])

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
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"\t[ reflect :: Error for User {user_id}, Cluster {cluster_id} :: {e} ]")

            # Add unaddressed arguments' scores
            for cluster_id, cluster in clusters[user_id].items():
                if cluster_id not in weight_mods:
                    try:
                        persuasiveness_score = float(
                            json.loads(cluster.wepcc_result['persuasiveness_justification'])
                            ['content']['persuasiveness_score'])

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
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"\t[ reflect :: Error for User {user_id}, Cluster {cluster_id} :: {e} ]")

            aggregated_scores[user_id] = total_score
            user_result["total_score"] = total_score
            results.append(user_result)
            print(f"\t[ reflect :: Aggregated score for User {user_id} :: {total_score} ]")

        # Process remaining clusters without shadow coverage
        for user_id, user_clusters in clusters.items():
            if user_id not in aggregated_scores:
                total_score = 0
                addressed_clusters[user_id] = []
                unaddressed_clusters[user_id] = []

                user_result = {"user": user_id, "clusters": []}

                for cluster_id, cluster in user_clusters.items():
                    if cluster_id not in cluster_shadow_coverage.get(user_id, {}):
                        try:
                            persuasiveness_score = float(
                                json.loads(cluster.wepcc_result['persuasiveness_justification'])['content'][
                                    'persuasiveness_score'])

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
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"\t[ reflect :: Error for User {user_id}, Cluster {cluster_id} :: {e} ]")

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

    def pretty_print_wepcc_result(self, user_id, cluster_id, wepcc_result):
        print(f"\t[ reflect :: WEPCC for user {user_id}, cluster {cluster_id} ]")

        print("\nWarrant:")
        print(json.loads(wepcc_result["warrant"])["content"])
        print("\nEvidence:")
        print(json.loads(wepcc_result["evidence"])["content"])
        print("\nPersuasiveness Justification:")
        print("Persuasiveness Score:", json.loads(wepcc_result["persuasiveness_justification"])["content"]["persuasiveness_score"])
        print("Justification:", json.loads(wepcc_result["persuasiveness_justification"])["content"]["justification"])
        print("\nClaim:")
        print(json.loads(wepcc_result["claim"])["content"])
        print("\nCounterclaim:")
        print(json.loads(wepcc_result["counterclaim"])["content"])
        print("\n")

