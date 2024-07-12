# argument_detection.py

import json
import logging
from collections import defaultdict

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from topos.FC.cache_manager import CacheManager
from topos.FC.similitude_module import load_model, util


class ArgumentDetection:
    def __init__(self, api_key, model="ollama:solar", max_tokens_warrant=250, max_tokens_evidence=250,
                 max_tokens_persuasiveness_justification=250, max_tokens_claim=250, max_tokens_counter_claim=500,
                 cache_enabled=True):
        self.api_key = api_key
        self.model_provider, self.model_type = self.parse_model(model)

        self.api_url = "unknown_api_url"
        if self.model_provider == "ollama":
            self.api_url = "http://localhost:11434/v1"
        elif self.model_provider == "openai":
            self.api_url = "http://localhost:3000/v1"
        elif self.model_provider == "claude":
            self.api_url = "http://localhost:3000/v1"

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
        return load_model(self.embedding_model_medium_420_2800)

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
        print(f'cluster_sentences:\n{cluster_sentences}\n')
        # returns
        # 1. the warrant
        # 3. the evidence
        # 6. the persuasiveness / justification
        # 2. the claim
        # 4. the counterclaim

        word_max_warrant = 300
        word_max_evidence = 300
        word_max_persuasiveness_justification = 150
        word_max_claim = 200
        word_max_counter_claim = 300

        warrant = self.fetch_argument_warrant(cluster_sentences, word_max_warrant, extra_fingerprint, max_retries=50)
        evidence = self.fetch_argument_evidence(cluster_sentences, word_max_evidence, extra_fingerprint, max_retries=50)
        # @note: this re-rolls because it needs to become quantized - a clipped mean would probably be best here.
        persuasiveness_justification = self.fetch_argument_persuasiveness_justification(cluster_sentences, word_max_persuasiveness_justification, extra_fingerprint, max_retries=50)
        claim = self.fetch_argument_claim(cluster_sentences, word_max_claim, extra_fingerprint, max_retries=50)
        counterclaim = self.fetch_argument_counter_claim(cluster_sentences, word_max_counter_claim, extra_fingerprint, max_retries=50)

        return warrant.content, evidence.content, persuasiveness_justification.content, claim.content, counterclaim.content

    def fetch_argument_warrant(self, cluster_sentences, word_max, extra_fingerprint="", max_retries=3):
        content_string = ""

        # if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
        content_string = f"""           Expected output: {{\"role\": \"warrant\", \"content\": \"_summary of warrant here, {word_max} max words, try to get as close to the max words as possible_\"}} 
                                        Instructions: Given the following cluster of sentences, [the warrant: identify the underlying reasoning or assumption that connects the evidence to the claim]. In the exact format below, provide a concise summary of the warrant only, no preamble. No negative constructions. 
                                        [user will enter data like]
                                        Cluster: 
                                        {{cluster_sentences}}
                                        """

        messages = [
            {
                "role": "system",
                "content": content_string
            }
        ]

        # if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
        messages.append({"role": "user",
                         "content": f""" Cluster: 
                                        {cluster_sentences}"""})
        temperature = 0.3
        formatted_json = json.dumps(messages, indent=4)

        content_key = self.get_content_key(formatted_json + extra_fingerprint, self.max_tokens_warrant)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            try:
                found = json.loads(cached_response.content)
                warrant_len = 1 / len(found['content'])  # this will raise an error if the JSON is invalid
                return cached_response
            except json.JSONDecodeError as json_err:
                logging.warning(f"JSONDecodeError on cached response: {json_err}")
            except ValueError as value_err:
                logging.warning(f"ValueError on cached response: {value_err}")
            except ZeroDivisionError as zero_err:
                logging.warning(f"ZeroDivisionError on cached response: {zero_err}")

        client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
        )

        cur_message = ""
        cur_response_content = ""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_type,
                    messages=json.loads(formatted_json),
                    max_tokens=self.max_tokens_warrant,
                    n=1,
                    stop=None,
                    temperature=temperature)

                response_content = response.choices[0].message

                cur_message = json.loads(formatted_json)
                cur_response_content = response_content

                found = json.loads(response_content.content)
                warrant_len = 1 / len(found['content'])

                self.cache_manager.save_to_cache(content_key, response_content)
                return response_content

            except json.JSONDecodeError as json_err:
                # print(f"cur_message: {cur_message}")
                print(f"warrant response: {cur_response_content.content}")
                logging.warning(f"JSONDecodeError on attempt {attempt + 1}/{max_retries}: {json_err}")
                continue

            except ValueError as value_err:
                print(f"warrant response: {cur_response_content.content}")
                logging.warning(f"ValueError on attempt {attempt + 1}/{max_retries}: {value_err}")
                continue

            except ZeroDivisionError as zero_err:
                logging.warning(f"ZeroDivisionError on attempt {attempt + 1}/{max_retries}: {zero_err}")
                continue

            except Exception as e:
                logging.error(f"Error in fetch_argument_warrant: {e}")
                break

        logging.error(f"Failed to fetch valid argument warrant after {max_retries} attempts.")
        return None

    def fetch_argument_evidence(self, cluster_sentences, word_max, extra_fingerprint="", max_retries=3):
        content_string = ""

        # if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
        content_string = f"""           Expected output: {{\"role\": \"evidence\", \"content\": \"_summary of evidence here, {word_max} max words, try to get as close to the max words as possible_\"}}
                                        Given the following cluster of sentences, [the evidence: identify the pieces of evidence that support the claim]. In the exact format below, provide a concise summary of the evidence only, no preamble. No negative constructions.
                                        [user will enter data like]
                                        Cluster: 
                                        {{cluster_sentences}}
                                        """

        messages = [
            {
                "role": "system",
                "content": content_string
            }
        ]

        # if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
        messages.append({"role": "user",
                         "content": f""" Cluster: 
                                        {cluster_sentences}"""})
        temperature = 0.3
        formatted_json = json.dumps(messages, indent=4)

        content_key = self.get_content_key(formatted_json + extra_fingerprint, self.max_tokens_evidence)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            try:
                found = json.loads(cached_response.content)
                evidence_len = 1 / len(found['content'])  # this will raise an error if the JSON is invalid
                return cached_response
            except json.JSONDecodeError as json_err:
                logging.warning(f"JSONDecodeError on cached response: {json_err}")
            except ValueError as value_err:
                logging.warning(f"ValueError on cached response: {value_err}")
            except ZeroDivisionError as zero_err:
                logging.warning(f"ZeroDivisionError on cached response: {zero_err}")

        client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
        )

        cur_message = ""
        cur_response_content = ""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_type,
                    messages=json.loads(formatted_json),
                    max_tokens=self.max_tokens_evidence,
                    n=1,
                    stop=None,
                    temperature=temperature)

                response_content = response.choices[0].message

                cur_message = json.loads(formatted_json)
                cur_response_content = response_content

                found = json.loads(response_content.content)
                evidence_len = 1 / len(found['content'])

                self.cache_manager.save_to_cache(content_key, response_content)
                return response_content

            except json.JSONDecodeError as json_err:
                # print(f"cur_message: {cur_message}")
                print(f"evidence response: {cur_response_content.content}")
                logging.warning(f"JSONDecodeError on attempt {attempt + 1}/{max_retries}: {json_err}")
                continue

            except ValueError as value_err:
                logging.warning(f"ValueError on attempt {attempt + 1}/{max_retries}: {value_err}")
                continue

            except ZeroDivisionError as zero_err:
                logging.warning(f"ZeroDivisionError on attempt {attempt + 1}/{max_retries}: {zero_err}")
                continue

            except Exception as e:
                logging.error(f"Error in fetch_argument_evidence: {e}")
                break

        logging.error(f"Failed to fetch valid argument evidence after {max_retries} attempts.")
        return None


    def fetch_argument_persuasiveness_justification(self, cluster_sentences, word_max, extra_fingerprint="",
                                                    max_retries=3):

        content_string = ""

        # if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
        content_string = f"""Given the following cluster of sentences, evaluate the persuasiveness of the arguments presented only, no preamble. No negative constructions.
                                        [user will enter data like]
                                        Cluster: 
                                        {{cluster_sentences}}
                                        [your output response-json (include braces) should be of the form]
                                        {{\"role\": \"persuasiveness\", \"content\": {{\"persuasiveness_score\": \"_1-10 integer here_\", \"justification\": \"_summary of justification here, {word_max} max words, try to get as close to the max words as possible_\" }} }}"""

        messages = [
            {
                "role": "system",
                "content": content_string
            }
        ]

        # if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
        messages.append({"role": "user",
                         "content": f""" Cluster: 
                                        {cluster_sentences}"""})
        temperature = 0.3
        formatted_json = json.dumps(messages, indent=4)

        content_key = self.get_content_key(formatted_json + extra_fingerprint,
                                           self.max_tokens_persuasiveness_justification)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            # Validate the JSON format
            try:
                found = json.loads(cached_response.content)  # This will raise an error if the JSON is invalid
                persuasiveness_score = float(found['content']['persuasiveness_score'])  # this will also raise an error

                return cached_response
            except json.JSONDecodeError as json_err:
                logging.warning(f"JSONDecodeError on cached response: {json_err}")
            except ValueError as value_err:
                logging.warning(f"ValueError on cached response: {value_err}")


        client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
        )

        cur_message = ""
        cur_response_content = ""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_type,
                    messages=json.loads(formatted_json),
                    max_tokens=self.max_tokens_persuasiveness_justification,
                    n=1,
                    stop=None,
                    temperature=temperature)

                response_content = response.choices[0].message

                # for debug/refinement of the prompt
                cur_message = json.loads(formatted_json)
                cur_response_content = response_content

                # Validate the JSON format
                # print(f"response: {response_content.content}")
                found = json.loads(response_content.content)  # This will raise an error if the JSON is invalid
                persuasiveness_score = float(found['content']['persuasiveness_score'])  # this will also raise an error

                self.cache_manager.save_to_cache(content_key, response_content)
                return response_content

            except json.JSONDecodeError as json_err:
                # print(f"cur_message: {cur_message}")
                print(f"persuasiveness/justification response: {cur_response_content.content}")
                logging.warning(f"JSONDecodeError on attempt {attempt + 1}/{max_retries}: {json_err}")
                continue  # Retry on JSON decode error

            except ValueError as value_err:
                logging.warning(f"ValueError on attempt {attempt + 1}/{max_retries}: {value_err}")
                continue  # Retry on Value error

            except Exception as e:
                logging.error(f"Error in fetch_argument_persuasiveness_justification: {e}")
                break  # Break on other exceptions

        logging.error(f"Failed to fetch valid argument persuasiveness justification after {max_retries} attempts.")
        return None

    def fetch_argument_claim(self, cluster_sentences, word_max, extra_fingerprint="", max_retries=3):
        content_string = ""

        # if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
        content_string = f"""           Expected output: {{\"role\": \"claim\", \"content\": \"_summary of claim here, {word_max} max words, try to get as close to the max words as possible_\"}}
                                        Given the following cluster of sentences, [the claim: identify the main claim or assertion made]. In the exact format below, provide a concise summary of the claim only, no preamble. No negative constructions. 150 words or less.
                                        [user will enter data like]
                                        Cluster: 
                                        {{cluster_sentences}}
                                        """

        messages = [
            {
                "role": "system",
                "content": content_string
            }
        ]

        # if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
        messages.append({"role": "user",
                         "content": f""" Cluster: 
                                        {cluster_sentences}"""})
        temperature = 0.3
        formatted_json = json.dumps(messages, indent=4)

        content_key = self.get_content_key(formatted_json + extra_fingerprint, self.max_tokens_claim)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            try:
                found = json.loads(cached_response.content)
                claim_len = 1 / len(found['content'])  # this will raise an error if the JSON is invalid
                return cached_response
            except json.JSONDecodeError as json_err:
                logging.warning(f"JSONDecodeError on cached response: {json_err}")
            except ValueError as value_err:
                logging.warning(f"ValueError on cached response: {value_err}")
            except ZeroDivisionError as zero_err:
                logging.warning(f"ZeroDivisionError on cached response: {zero_err}")


        client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
        )

        cur_message = ""
        cur_response_content = ""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_type,
                    messages=json.loads(formatted_json),
                    max_tokens=self.max_tokens_claim,
                    n=1,
                    stop=None,
                    temperature=temperature)

                response_content = response.choices[0].message

                cur_message = json.loads(formatted_json)
                cur_response_content = response_content

                found = json.loads(response_content.content)
                claim_len = 1 / len(found['content'])

                self.cache_manager.save_to_cache(content_key, response_content)
                return response_content

            except json.JSONDecodeError as json_err:
                # print(f"cur_message: {cur_message}")
                print(f"claim response: {cur_response_content.content}")
                logging.warning(f"JSONDecodeError on attempt {attempt + 1}/{max_retries}: {json_err}")
                continue

            except ValueError as value_err:
                logging.warning(f"ValueError on attempt {attempt + 1}/{max_retries}: {value_err}")
                continue

            except ZeroDivisionError as zero_err:
                logging.warning(f"ZeroDivisionError on attempt {attempt + 1}/{max_retries}: {zero_err}")
                continue

            except Exception as e:
                logging.error(f"Error in fetch_argument_claim: {e}")
                break

        logging.error(f"Failed to fetch valid argument claim after {max_retries} attempts.")
        return None

    def fetch_argument_counter_claim(self, cluster_sentences, word_max, extra_fingerprint="", max_retries=3):
        content_string = ""

        # if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
        content_string = f"""           Expected output: {{\"role\": \"counter_claim\", \"content\": \"_summary of counter claim here, {word_max} max words, try to get as close to the max words as possible!_\"}}
                                        Given the following cluster of sentences, [the counterclaim: identify any counterclaims or opposing arguments presented]. In the exact format below, provide a concise summary of the counterclaims only, no preamble. No negative constructions.
                                        [user will enter data like]
                                        Cluster: 
                                        {{cluster_sentences}}
                           """
        messages = [
            {
                "role": "system",
                "content": content_string
            }
        ]

        # if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
        messages.append({"role": "user",
                         "content": f""" Cluster: 
                                        {cluster_sentences}"""})
        temperature = 0.3
        formatted_json = json.dumps(messages, indent=4)

        content_key = self.get_content_key(formatted_json + extra_fingerprint, self.max_tokens_counter_claim)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            try:
                found = json.loads(cached_response.content)
                counter_claim_len = 1 / len(found['content'])  # this will raise an error if the JSON is invalid
                return cached_response
            except json.JSONDecodeError as json_err:
                logging.warning(f"JSONDecodeError on cached response: {json_err}")
            except ValueError as value_err:
                logging.warning(f"ValueError on cached response: {value_err}")
            except ZeroDivisionError as zero_err:
                logging.warning(f"ZeroDivisionError on cached response: {zero_err}")


        client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
        )

        cur_message = ""
        cur_response_content = ""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_type,
                    messages=json.loads(formatted_json),
                    max_tokens=self.max_tokens_counter_claim,
                    n=1,
                    stop=None,
                    temperature=temperature)

                response_content = response.choices[0].message

                cur_message = json.loads(formatted_json)
                cur_response_content = response_content

                found = json.loads(response_content.content)
                counter_claim_len = 1 / len(found['content'])  # this will raise an error if the JSON is invalid

                self.cache_manager.save_to_cache(content_key, response_content)
                return response_content

            except json.JSONDecodeError as json_err:
                # print(f"cur_message: {cur_message}")
                print(f"counterclaim response: {cur_response_content.content}")
                logging.warning(f"JSONDecodeError on attempt {attempt + 1}/{max_retries}: {json_err}")
                continue

            except ValueError as value_err:
                logging.warning(f"ValueError on attempt {attempt + 1}/{max_retries}: {value_err}")
                continue

            except ZeroDivisionError as zero_err:
                logging.warning(f"ZeroDivisionError on attempt {attempt + 1}/{max_retries}: {zero_err}")
                continue

            except Exception as e:
                logging.error(f"Error in fetch_argument_counter_claim: {e}")
                break

        logging.error(f"Failed to fetch valid argument counter claim after {max_retries} attempts.")
        return None

    def get_embeddings(self, sentences):
        # print("[INFO] Embedding sentences using SentenceTransformer...")
        embeddings = self.model.encode(sentences)
        # print("[INFO] Sentence embeddings obtained.")
        return embeddings

    from sklearn.cluster import AgglomerativeClustering
    from sentence_transformers import SentenceTransformer
    import numpy as np

    def cluster_sentences(self, sentences, distance_threshold=0.5):
        print("\t\t[ [INFO] Performing hierarchical clustering... ]")
        embeddings = self.get_embeddings(sentences)

        # Perform Agglomerative Clustering based on cosine similarity
        clustering = AgglomerativeClustering(n_clusters=None,
                                             distance_threshold=distance_threshold,
                                             metric='cosine',
                                             linkage='average')
        clusters = clustering.fit_predict(embeddings)

        print("\t\t[ [INFO] Clustering complete. Clusters assigned ]")
        for i, cluster in enumerate(clusters):
            print(f"\t\t\t[ Sentence {i + 1} is in cluster {cluster} ]")

        cluster_dict = {}
        coherence_scores = {}
        for i, cluster in enumerate(clusters):
            if cluster not in cluster_dict:
                cluster_dict[cluster] = []
            cluster_dict[cluster].append(sentences[i])

        # Calculate coherence for each cluster
        for cluster, cluster_sentences in cluster_dict.items():
            cluster_embeddings = self.get_embeddings(cluster_sentences)
            coherence = self.calculate_coherence(cluster_embeddings)
            coherence_scores[cluster] = float(coherence)
            # print(f"\t\t\t[ Cluster {cluster} coherence: {coherence:.4f} ]")

        return cluster_dict, coherence_scores

    @staticmethod
    def calculate_coherence(embeddings):
        similarity_matrix = cosine_similarity(embeddings)
        return np.mean(similarity_matrix)

    # def calculate_distance_matrix(self, embeddings):
    #     print("[INFO] Calculating semantic distance matrix...")
    #     # distance_matrix = np.zeros((len(embeddings), len(embeddings)))
    #     # for i in range(len(embeddings)):
    #     #     for j in range(len(embeddings)):
    #     #         if i != j:
    #     #             distance_matrix[i][j] = np.linalg.norm(embeddings[i] - embeddings[j])
    #
    #     # Use pdist to calculate the condensed distance matrix
    #     distance_matrix = pdist(embeddings, metric='euclidean')
    #
    #     print("[INFO] Distance matrix calculated.")
    #     return distance_matrix
    #
    # def calculate_distance_matrix_square(self, embeddings):
    #     print("\t\t\t[ [INFO] Calculating semantic distance matrix... ]")
    #     distance_matrix = np.zeros((len(embeddings), len(embeddings)))
    #     for i in range(len(embeddings)):
    #         for j in range(len(embeddings)):
    #             if i != j:
    #                 distance_matrix[i][j] = np.linalg.norm(embeddings[i] - embeddings[j])
    #
    #     print("\t\t\t[ [INFO] Distance matrix calculated. ]")
    #     return distance_matrix
    #
    # def cluster_sentences(self, sentences, distance_threshold=1.5):  # Adjust distance_threshold here
    #     print("\t\t[ [INFO] Performing hierarchical clustering... ]")
    #     embeddings = self.get_embeddings(sentences)
    #     distance_matrix = self.calculate_distance_matrix_square(embeddings)
    #     # distance_matrix = self.calculate_distance_matrix(embeddings)
    #
    #     # Perform Agglomerative Clustering based on the distance matrix
    #     clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold,
    #                                          metric='euclidean', linkage='average')
    #     clusters = clustering.fit_predict(distance_matrix)
    #
    #     # @note: @jonny - this might be better - testing
    #     # clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold,
    #     #                                      metric='precomputed', linkage='average')
    #     # Convert condensed distance matrix back to a full square form for clustering
    #     # full_distance_matrix = squareform(distance_matrix)
    #     # clusters = clustering.fit_predict(full_distance_matrix)
    #
    #     print("\t\t[ [INFO] Clustering complete. Clusters assigned ]")
    #     for i, cluster in enumerate(clusters):
    #         print(f"\t\t\t[ Sentence {i + 1} is in cluster {cluster} ]")
    #
    #     cluster_dict = {}
    #     for i, cluster in enumerate(clusters):
    #         if cluster not in cluster_dict:
    #             cluster_dict[cluster] = []
    #         cluster_dict[cluster].append(sentences[i])
    #
    #     return cluster_dict

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
