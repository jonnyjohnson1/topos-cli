import json
import logging


from sklearn.cluster import AgglomerativeClustering
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from topos.FC.cache_manager import CacheManager
from topos.FC.similitude_module import load_model, util


class ArgumentDetection:
    def __init__(self, api_key, model="ollama:solar", max_tokens_warrant=128, max_tokens_evidence=128,
                 max_tokens_persuasiveness_justification=128, max_tokens_claim=128, max_tokens_counter_claim=128,
                 cache_enabled=True):
        self.api_key = api_key
        self.model_provider, self.model_type = self.parse_model(model)
        self.max_tokens_warrant = max_tokens_warrant
        self.max_tokens_evidence = max_tokens_evidence
        self.max_tokens_persuasiveness_justification = max_tokens_persuasiveness_justification
        self.max_tokens_claim = max_tokens_claim
        self.max_tokens_counter_claim = max_tokens_counter_claim
        self.cache_enabled = cache_enabled

        self.embedding_model_smallest_80_14200 = 'all-MiniLM-L6-v2'
        self.embedding_model_small_120_7500 = 'all-MiniLM-L12-v2'
        self.embedding_model_medium_420_2800 = 'all-mpnet-base-v2'

        self.model = self.load_model()

        self.cache_manager = CacheManager()

    def load_model(self):
        return load_model(self.embedding_model_smallest_80_14200)

    @staticmethod
    def parse_model(model):
        if ":" in model:
            return model.split(":", 1)
        else:
            return "ollama", model

    def get_content_key(self, key, token_limit_for_task):
        content_key = f"{key}.{self.model_provider}.{self.model_type}.{token_limit_for_task}"
        return content_key

    def fetch_argument_definition(self, cluster_sentences, extra_fingerprint=""):
        # returns
        # 1. the warrant
        # 3. the evidence
        # 6. the persuasiveness / justification
        # 2. the claim
        # 4. the counterclaim

        word_max_warrant = 30
        word_max_evidence = 50
        word_max_persuasiveness_justification = 30
        word_max_claim = 20
        word_max_counter_claim = 30

        warrant = self.fetch_argument_warrant(cluster_sentences, word_max_warrant, extra_fingerprint="")
        evidence = self.fetch_argument_evidence(cluster_sentences, word_max_evidence, extra_fingerprint="")
        persuasiveness_justification = self.fetch_argument_persuasiveness_justification(cluster_sentences, word_max_persuasiveness_justification, extra_fingerprint="")
        claim = self.fetch_argument_claim(cluster_sentences, word_max_claim, extra_fingerprint="")
        counterclaim = self.fetch_argument_counter_claim(cluster_sentences, word_max_counter_claim, extra_fingerprint="")

        return warrant.content, evidence.content, persuasiveness_justification.content, claim.content, counterclaim.content

    def fetch_argument_warrant(self, cluster_sentences, word_max, extra_fingerprint=""):

        content_string = ""

        if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            content_string = f"""Given the following cluster of sentences, identify the underlying reasoning or assumption that connects the evidence to the claim. Provide a concise summary of the warrant.
                                            [user will enter data like]
                                            Cluster: 
                                            {{cluster_sentences}}
                                            [your output response-json should be of the form]
                                            {{\"role\": \"warrant\", \"content\": \"_summary of warrant here, {word_max} max words!_\"}}"""

        # Construct the JSON object using a Python dictionary and convert it to a JSON string
        messages = [
            {
                "role": "system",
                "content": content_string
            }
        ]

        if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            messages.append({"role": "user",
                             "content": f""" Cluster: 
                                            {cluster_sentences}"""})
        # default temp is 0.3
        temperature = 0.3

        # Use json.dumps to safely create a JSON string
        # Attempt to parse the template as JSON
        formatted_json = json.dumps(messages, indent=4)

        content_key = self.get_content_key(formatted_json + extra_fingerprint, self.max_tokens_warrant)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            return cached_response

        try:
            ollama_base = "http://localhost:11434/v1"
            client = OpenAI(
                base_url=ollama_base,
                api_key="ollama",
            )

            response = client.chat.completions.create(
                model=self.model_type,
                messages=json.loads(formatted_json),
                max_tokens=self.max_tokens_warrant,
                n=1,
                stop=None,
                temperature=temperature)
            self.cache_manager.save_to_cache(content_key, response.
                                             choices[0].message)
            return response.choices[0].message
        except Exception as e:
            logging.error(f"Error in fetch_argument_warrant: {e}")
            return None

    def fetch_argument_evidence(self, cluster_sentences, word_max, extra_fingerprint=""):

        content_string = ""

        if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            content_string = f"""Given the following cluster of sentences, identify the pieces of evidence that support the claim. Provide a concise summary of the evidence.
                                            [user will enter data like]
                                            Cluster: 
                                            {{cluster_sentences}}
                                            [your output response-json should be of the form]
                                            {{\"role\": \"evidence\", \"content\": \"_summary of evidence here, {word_max} max words!_\"}}"""

        # Construct the JSON object using a Python dictionary and convert it to a JSON string
        messages = [
            {
                "role": "system",
                "content": content_string
            }
        ]

        if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            messages.append({"role": "user",
                             "content": f""" Cluster: 
                                            {cluster_sentences}"""})
        # default temp is 0.3
        temperature = 0.3

        # Use json.dumps to safely create a JSON string
        # Attempt to parse the template as JSON
        formatted_json = json.dumps(messages, indent=4)

        content_key = self.get_content_key(formatted_json + extra_fingerprint, self.max_tokens_evidence)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            return cached_response

        try:
            ollama_base = "http://localhost:11434/v1"
            client = OpenAI(
                base_url=ollama_base,
                api_key="ollama",
            )

            response = client.chat.completions.create(
                model=self.model_type,
                messages=json.loads(formatted_json),
                max_tokens=self.max_tokens_evidence,
                n=1,
                stop=None,
                temperature=temperature)
            self.cache_manager.save_to_cache(content_key, response.
                                             choices[0].message)
            return response.choices[0].message
        except Exception as e:
            logging.error(f"Error in fetch_argument_warrant: {e}")
            return None

    def fetch_argument_persuasiveness_justification(self, cluster_sentences, word_max, extra_fingerprint=""):

        content_string = ""

        if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            content_string = f"""Given the following cluster of sentences, evaluate the persuasiveness of the arguments presented. Rate the persuasiveness on a scale from 1 to 10 and provide a brief justification.
                                            [user will enter data like]
                                            Cluster: 
                                            {{cluster_sentences}}
                                            [your output response-json should be of the form]
                                            {{\"role\": \"persuasiveness\", \"content\": {{\"persuasiveness_score\": \"_1-10 integer here_\", \"justification\": \"_summary of justification here, {word_max} max words!_\"}}}}"""

        # Construct the JSON object using a Python dictionary and convert it to a JSON string
        messages = [
            {
                "role": "system",
                "content": content_string
            }
        ]

        if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            messages.append({"role": "user",
                             "content": f""" Cluster: 
                                            {cluster_sentences}"""})
        # default temp is 0.3
        temperature = 0.3

        # Use json.dumps to safely create a JSON string
        # Attempt to parse the template as JSON
        formatted_json = json.dumps(messages, indent=4)

        content_key = self.get_content_key(formatted_json + extra_fingerprint, self.max_tokens_persuasiveness_justification)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            return cached_response

        try:
            ollama_base = "http://localhost:11434/v1"
            client = OpenAI(
                base_url=ollama_base,
                api_key="ollama",
            )

            response = client.chat.completions.create(
                model=self.model_type,
                messages=json.loads(formatted_json),
                max_tokens=self.max_tokens_persuasiveness_justification,
                n=1,
                stop=None,
                temperature=temperature)
            self.cache_manager.save_to_cache(content_key, response.
                                             choices[0].message)
            return response.choices[0].message
        except Exception as e:
            logging.error(f"Error in fetch_argument_persuasiveness_justification: {e}")
            return None

    def fetch_argument_claim(self, cluster_sentences, word_max, extra_fingerprint=""):

        content_string = ""

        if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            content_string = f"""Given the following cluster of sentences, identify the main claim or assertion made. Provide a concise summary of the claim.
                                            [user will enter data like]
                                            Cluster: 
                                            {{cluster_sentences}}
                                            [your output response-json should be of the form]
                                            {{\"role\": \"claim\", \"content\": \"_summary of claim here, {word_max} max words!_\"}}"""

        # Construct the JSON object using a Python dictionary and convert it to a JSON string
        messages = [
            {
                "role": "system",
                "content": content_string
            }
        ]

        if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            messages.append({"role": "user",
                             "content": f""" Cluster: 
                                            {cluster_sentences}"""})
        # default temp is 0.3
        temperature = 0.3

        # Use json.dumps to safely create a JSON string
        # Attempt to parse the template as JSON
        formatted_json = json.dumps(messages, indent=4)

        content_key = self.get_content_key(formatted_json + extra_fingerprint, self.max_tokens_claim)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            return cached_response

        try:
            ollama_base = "http://localhost:11434/v1"
            client = OpenAI(
                base_url=ollama_base,
                api_key="ollama",
            )

            response = client.chat.completions.create(
                model=self.model_type,
                messages=json.loads(formatted_json),
                max_tokens=self.max_tokens_claim,
                n=1,
                stop=None,
                temperature=temperature)
            self.cache_manager.save_to_cache(content_key, response.
                                             choices[0].message)
            return response.choices[0].message
        except Exception as e:
            logging.error(f"Error in fetch_argument_claim: {e}")
            return None

    def fetch_argument_counter_claim(self, cluster_sentences, word_max, extra_fingerprint=""):

        content_string = ""

        if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            content_string = f"""Given the following cluster of sentences, identify any counterclaims or opposing arguments presented. Provide a concise summary of the counterclaims.
                                            [user will enter data like]
                                            Cluster: 
                                            {{cluster_sentences}}
                                            [your output response-json should be of the form]
                                            {{\"role\": \"counter_claim\", \"content\": \"_summary of counter claim here, {word_max} max words!_\"}}"""

        # Construct the JSON object using a Python dictionary and convert it to a JSON string
        messages = [
            {
                "role": "system",
                "content": content_string
            }
        ]

        if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            messages.append({"role": "user",
                             "content": f""" Cluster: 
                                            {cluster_sentences}"""})
        # default temp is 0.3
        temperature = 0.3

        # Use json.dumps to safely create a JSON string
        # Attempt to parse the template as JSON
        formatted_json = json.dumps(messages, indent=4)

        content_key = self.get_content_key(formatted_json + extra_fingerprint, self.max_tokens_counter_claim)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            return cached_response

        try:
            ollama_base = "http://localhost:11434/v1"
            client = OpenAI(
                base_url=ollama_base,
                api_key="ollama",
            )

            response = client.chat.completions.create(
                model=self.model_type,
                messages=json.loads(formatted_json),
                max_tokens=self.max_tokens_claim,
                n=1,
                stop=None,
                temperature=temperature)
            self.cache_manager.save_to_cache(content_key, response.
                                             choices[0].message)
            return response.choices[0].message
        except Exception as e:
            logging.error(f"Error in fetch_argument_counter_claim: {e}")
            return None

    def get_embeddings(self, sentences):
        print("[INFO] Embedding sentences using SentenceTransformer...")
        embeddings = self.model.encode(sentences)
        print("[INFO] Sentence embeddings obtained.")
        return embeddings

    def calculate_distance_matrix(self, embeddings):
        print("[INFO] Calculating semantic distance matrix...")
        distance_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(embeddings[i] - embeddings[j])
        print("[INFO] Distance matrix calculated.")
        return distance_matrix

    def cluster_sentences(self, sentences, distance_threshold=1.5):  # Adjust distance_threshold here
        embeddings = self.get_embeddings(sentences)
        distance_matrix = self.calculate_distance_matrix(embeddings)

        # Perform Agglomerative Clustering based on the distance matrix
        print("[INFO] Performing hierarchical clustering...")
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric='euclidean', linkage='average')
        clusters = clustering.fit_predict(distance_matrix)

        print("[INFO] Clustering complete. Clusters assigned:")
        for i, cluster in enumerate(clusters):
            print(f"Sentence {i + 1} is in cluster {cluster}")

        cluster_dict = {}
        for i, cluster in enumerate(clusters):
            if cluster not in cluster_dict:
                cluster_dict[cluster] = []
            cluster_dict[cluster].append(sentences[i])

        return cluster_dict

    def run_tests(self):
        examples = {
            "Debate": [
                "Social media platforms have become the primary source of information for many people.",
                "They have the power to influence public opinion and election outcomes.",
                "Government regulation could help in mitigating the spread of false information.",
                "On the other hand, government intervention might infringe on freedom of speech.",
                "Social media companies are already taking steps to address misinformation.",
                "Self-regulation is preferable as it avoids the risks of government overreach.",
                "The lack of regulation has led to the proliferation of harmful content and echo chambers."
            ],
            "Chess": [
                "Chess is a game of deeper strategy compared to checkers.",
                "It offers a complexity that requires players to think several moves ahead, promoting strategic thinking and planning skills.",
                "Each piece in chess has its own unique moves and capabilities, unlike the uniform pieces in checkers, adding layers of strategy and tactics.",
                "Furthermore, chess has a rich history and cultural significance that checkers lacks.",
                "The game has been played by kings and commoners alike for centuries and has influenced various aspects of art, literature, and even politics.",
                "This cultural depth adds to the enjoyment and appreciation of the game.",
                "Chess also offers more varied and challenging gameplay.",
                "The opening moves alone in chess provide a nearly infinite number of possibilities, leading to different game progressions each time.",
                "Checkers, by contrast, has a more limited set of opening moves, which can make the game feel repetitive over time.",
                "Finally, chess is recognized globally as a competitive sport with international tournaments and rankings.",
                "This global recognition and the opportunities for competition at all levels make chess a more engaging and rewarding game for those who enjoy not only playing but also watching and studying the game."
            ],
            "Reading": [
                "Reading is a more engaging activity compared to watching.",
                "It stimulates the imagination and enhances cognitive functions in ways that watching cannot.",
                "Books often provide a deeper understanding of characters and plot, allowing for a more immersive experience.",
                "Furthermore, reading improves vocabulary and language skills, which is not as effectively achieved through watching.",
                "Reading also promotes better concentration and focus, as it requires active participation from the reader.",
                "Finally, reading is a more personal experience, allowing individuals to interpret and visualize the story in their own unique way."
            ]
        }

        for topic, sentences in examples.items():
            print(f"[INFO] Running test for: {topic}")
            clusters = self.cluster_sentences(sentences, distance_threshold=1.45)  # Adjust the threshold value here
            print(f"[INFO] Final Clusters for {topic}:")
            for cluster_id, cluster_sentences in clusters.items():
                print(f"Cluster {cluster_id}:")
                for sentence in cluster_sentences:
                    print(f"  - {sentence}")
            print("-" * 80)

#
# # Example usage
# argument_detection = ArgumentDetection()
# argument_detection.run_tests()
