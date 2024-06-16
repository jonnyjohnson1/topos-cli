import numpy as np
from sentence_transformers import CrossEncoder
from scipy.cluster.hierarchy import linkage, fcluster
from topos.FC.semantic_compression import SemanticCompression

class ArgumentDetection:
    def __init__(self, api_key):
        model = "dolphin-llama3"
        self.semantic_compression = SemanticCompression(api_key=api_key, model=f"ollama:{model}")

    def get_semantic_category(self, sentence):
        print(f"[INFO] Fetching semantic category for sentence: {sentence[:30]}...")
        category = self.semantic_compression.fetch_semantic_category(sentence)
        print(f"[INFO] Semantic category: {category.content}")
        return category.content

    def calculate_semantic_distance(self, summary1, summary2):
        distance = self.semantic_compression.get_semantic_distance(summary1, summary2)
        print(f"[INFO] Calculating distance between \"{summary1}\" and \"{summary2}\" -> Distance: {distance}")
        return distance

    def calculate_similarity_scores(self, sentences):
        print("[INFO] Calculating similarity scores using CrossEncoder...")
        model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
        pairs = [(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]
        scores = model.predict(pairs)
        print("[INFO] Similarity scores obtained:")
        for i, score in enumerate(scores):
            print(f"[INFO] Pair {i + 1}: \"{pairs[i][0][:30]}...\" <-> \"{pairs[i][1][:30]}...\" :: Score: {score}")
        return scores

    def cluster_sentences(self, sentences):
        print("[INFO] Creating summaries for sentences...")
        summaries = [self.get_semantic_category(sentence) for sentence in sentences]
        distance_matrix = np.zeros((len(summaries), len(summaries)))

        print("[INFO] Calculating semantic distance matrix...")
        for i in range(len(summaries)):
            for j in range(len(summaries)):
                if i != j:
                    distance_matrix[i][j] = self.calculate_semantic_distance(summaries[i], summaries[j])

        print("[INFO] Distance matrix calculated.")
        print(distance_matrix)

        print("[INFO] Calculating similarity scores for edge detection...")
        similarity_scores = self.calculate_similarity_scores(sentences)

        # Apply similarity score edge detection to adjust distances
        print("[INFO] Adjusting distances based on similarity scores...")
        for i in range(len(similarity_scores)):
            max_score = np.max(similarity_scores[i])
            if max_score < 0.5:  # Example threshold for low similarity
                distance_matrix[i][i + 1] += 2  # Increase distance if similarity score is low

        print("[INFO] Adjusted distance matrix:")
        print(distance_matrix)

        print("[INFO] Performing hierarchical clustering...")
        Z = linkage(distance_matrix, method='ward')
        clusters = fcluster(Z, t=0.8, criterion='distance')  # Adjusted threshold
        print("[INFO] Clustering complete. Clusters assigned:")
        for i, cluster in enumerate(clusters):
            print(f"Sentence {i + 1} is in cluster {cluster}")

        cluster_dict = {}
        for i, cluster in enumerate(clusters):
            if cluster not in cluster_dict:
                cluster_dict[cluster] = []
            cluster_dict[cluster].append(sentences[i])

        return cluster_dict

# Example usage:
if __name__ == "__main__":
    api_key = "your-api-key"
    argument_detector = ArgumentDetection(api_key)

    # Example sentences for the debate
    debate_sentences = [
        "Social media platforms have become the primary source of information for many people.",
        "They have the power to influence public opinion and election outcomes.",
        "Government regulation could help in mitigating the spread of false information.",
        "On the other hand, government intervention might infringe on freedom of speech.",
        "Social media companies are already taking steps to address misinformation.",
        "Self-regulation is preferable as it avoids the risks of government overreach.",
        "The lack of regulation has led to the proliferation of harmful content and echo chambers."
    ]

    clusters = argument_detector.cluster_sentences(debate_sentences)
    print("[INFO] Final Clusters:")
    for cluster_id, sentences in clusters.items():
        print(f"Cluster {cluster_id}:")
        for sentence in sentences:
            print(f"  - {sentence}")
