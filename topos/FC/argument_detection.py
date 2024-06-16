from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sentence_transformers import SentenceTransformer

class ArgumentDetection:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

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


# Example usage
argument_detection = ArgumentDetection()
argument_detection.run_tests()
