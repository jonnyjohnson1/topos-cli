import unittest
from sentence_transformers import CrossEncoder, SentenceTransformer
import sentencepiece  # Ensure sentencepiece is installed
import google.protobuf  # Ensure protobuf is installed
from nltk.tokenize import sent_tokenize
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Initialize the Cross-Encoder model
model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

class TestArgumentSegmentation(unittest.TestCase):

    def setUp(self):
        # Initialize the Cross-Encoder model
        self.model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Example 1: "Chess is better than checkers"
        self.chess_sentences = [
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
        ]

        # Example 2: "Reading is better than watching"
        self.reading_sentences = [
            "Reading is a more engaging activity compared to watching.",
            "It stimulates the imagination and enhances cognitive functions in ways that watching cannot.",
            "Books often provide a deeper understanding of characters and plot, allowing for a more immersive experience.",
            "Furthermore, reading improves vocabulary and language skills, which is not as effectively achieved through watching.",
            "Reading also promotes better concentration and focus, as it requires active participation from the reader.",
            "Finally, reading is a more personal experience, allowing individuals to interpret and visualize the story in their own unique way."
        ]

    def calculate_similarity(self, sentences):
        scores = []
        print("[ INFO :: Calculating similarity scores ]")
        for i in range(len(sentences) - 2):
            pair1 = (sentences[i], sentences[i+1])
            pair2 = (sentences[i+1], sentences[i+2])
            score1 = self.model.predict([pair1])[0]
            score2 = self.model.predict([pair2])[0]
            scores.append((score1, score2))
            print(f"\t[ Pair1: \"{pair1[0][:20]}...\" <-> \"{pair1[1][:20]}...\" :: Score1: {score1} ]")
            print(f"\t[ Pair2: \"{pair2[0][:20]}...\" <-> \"{pair2[1][:20]}...\" :: Score2: {score2} ]")
        return scores

    def assign_labels(self, scores):
        label_mapping = ['contradiction', 'entailment', 'neutral']
        labels = []
        print("[ INFO :: Assigning labels to scores ]")
        for score1, score2 in scores:
            label1 = label_mapping[score1.argmax()]
            label2 = label_mapping[score2.argmax()]
            labels.append((label1, label2))
            print(f"\t[ Score1: {score1} :: Label1: {label1} ]")
            print(f"\t[ Score2: {score2} :: Label2: {label2} ]")
        return labels

    def analyze_patterns(self, labels):
        boundaries = []
        print("[ INFO :: Analyzing patterns to determine boundaries ]")
        for i in range(len(labels) - 1):
            if labels[i][1] == 'entailment' and labels[i + 1][0] == 'contradiction':
                boundaries.append(i + 1)
                print(f"\t[ Boundary found at index: {i + 1} :: Pattern: entailment -> contradiction ]")
            elif labels[i][1] == 'contradiction' and labels[i + 1][0] == 'entailment':
                boundaries.append(i)
                print(f"\t[ Boundary found at index: {i} :: Pattern: contradiction -> entailment ]")
        return boundaries

    def refine_boundaries(self, sentences, boundaries):
        refined_boundaries = []
        current_segment = [sentences[0]]
        print("[ INFO :: Refining boundaries for final segmentation ]")
        for i in range(1, len(sentences)):
            if i in boundaries:
                refined_boundaries.append(current_segment)
                print(f"\t[ New segment created at boundary index: {i} :: Segment: {[s[:20] + '...' for s in current_segment]} ]")
                current_segment = []
            current_segment.append(sentences[i])
        refined_boundaries.append(current_segment)
        print(f"\t[ Final segment added: {[s[:20] + '...' for s in current_segment]} ]")
        return refined_boundaries

    def plot_similarity_scores(self, similarity_scores):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        score_labels = ['Contradiction', 'Entailment', 'Neutral']
        for i in range(3):
            axs[i].plot([score[i] for pair in similarity_scores for score in pair], marker='o')
            axs[i].set_title(f'Similarity Scores: {score_labels[i]}')
            axs[i].set_xlabel('Sentence Pair Index')
            axs[i].set_ylabel('Score')
        plt.tight_layout()
        plt.show()

    def plot_label_assignments(self, labels):
        label_counts = {'contradiction': 0, 'entailment': 0, 'neutral': 0}
        for label1, label2 in labels:
            label_counts[label1] += 1
            label_counts[label2] += 1
        plt.bar(label_counts.keys(), label_counts.values())
        plt.title('Label Assignments')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.show()

    def mark_boundaries(self, similarity_scores, boundaries):
        plt.plot([score[2] for pair in similarity_scores for score in pair], marker='o')
        for boundary in boundaries:
            plt.axvline(x=boundary, color='r', linestyle='--')
        plt.title('Similarity Scores with Boundaries')
        plt.xlabel('Sentence Pair Index')
        plt.ylabel('Neutral Score')
        plt.show()

    def plot_final_segments(self, segments):
        fig, ax = plt.subplots(figsize=(10, 5))
        segment_labels = [f'Segment {i+1}' for i in range(len(segments))]
        y_pos = np.arange(len(segment_labels))
        sentence_counts = [len(segment) for segment in segments]
        ax.barh(y_pos, sentence_counts, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(segment_labels)
        ax.invert_yaxis()
        ax.set_xlabel('Number of Sentences')
        ax.set_title('Final Segments')
        plt.show()

    def test_argument_segmentation(self):
        # Process both sets of sentences
        for sentences in [self.chess_sentences, self.reading_sentences]:
            print("[ INFO :: Starting new segmentation test ]")
            scores = self.calculate_similarity(sentences)
            labels = self.assign_labels(scores)
            boundaries = self.analyze_patterns(labels)
            segments = self.refine_boundaries(sentences, boundaries)
            print(f"[ INFO :: Final Segments: {[[s[:20] + '...' for s in segment] for segment in segments]} ]")
            print("-" * 80)

            # Plotting
            self.plot_similarity_scores(scores)
            self.plot_label_assignments(labels)
            self.mark_boundaries(scores, boundaries)
            self.plot_final_segments(segments)

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
