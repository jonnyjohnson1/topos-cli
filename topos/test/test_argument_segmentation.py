import unittest

from scipy.stats import hmean, trim_mean
from sentence_transformers import CrossEncoder, SentenceTransformer
import sentencepiece  # Ensure sentencepiece is installed
import google.protobuf  # Ensure protobuf is installed
from nltk.tokenize import sent_tokenize
import numpy as np

# # Determine threshold using the harmonic mean
# # threshold = hmean(differentiation_scores)
#
# # Determine threshold dynamically if needed
# # threshold = sum(differentiation_scores) / len(differentiation_scores)  # Example: average differentiation score
#
# # Calculate the trimmed mean, trimming 10% of the smallest and largest values
# threshold = trim_mean(differentiation_scores, 0.1)
#
#


# # Initialize the Cross-Encoder model
# # self.model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
# self.model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
#
#
# # self.model = CrossEncoder("cross-encoder/nli-roberta-large")
# # self.model = CrossEncoder("cross-encoder/nli-mpnet-base-v2")
# # self.model = CrossEncoder("cross-encoder/stsb-roberta-large")
# # self.model = CrossEncoder("cross-encoder/roberta-large")


class TestArgumentSegmentation(unittest.TestCase):

    def setUp(self):
        # Initialize the Cross-Encoder model
        self.model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        # self.model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
        # self.model = CrossEncoder("cross-encoder/nli-roberta-large")
        # self.model = CrossEncoder("cross-encoder/nli-mpnet-base-v2")
        # self.model = CrossEncoder("cross-encoder/stsb-roberta-large")
        # self.model = CrossEncoder("cross-encoder/roberta-large")

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

        self.chess_text = """Chess is a game of deeper strategy compared to checkers.
                    It offers a complexity that requires players to think several moves ahead, promoting strategic thinking and planning skills.
                    Each piece in chess has its own unique moves and capabilities, unlike the uniform pieces in checkers, adding layers of strategy and tactics.
                    Furthermore, chess has a rich history and cultural significance that checkers lacks.
                    The game has been played by kings and commoners alike for centuries and has influenced various aspects of art, literature, and even politics.
                    This cultural depth adds to the enjoyment and appreciation of the game.
                    Chess also offers more varied and challenging gameplay.
                    The opening moves alone in chess provide a nearly infinite number of possibilities, leading to different game progressions each time.
                    Checkers, by contrast, has a more limited set of opening moves, which can make the game feel repetitive over time.
                    Finally, chess is recognized globally as a competitive sport with international tournaments and rankings.
                    This global recognition and the opportunities for competition at all levels make chess a more engaging and rewarding game for those who enjoy not only playing but also watching and studying the game."""

        # Example 2: "Reading is better than watching"
        self.reading_sentences = [
            "Reading is a more engaging activity compared to watching.",
            "It stimulates the imagination and enhances cognitive functions in ways that watching cannot.",
            "Books often provide a deeper understanding of characters and plot, allowing for a more immersive experience.",
            "Furthermore, reading improves vocabulary and language skills, which is not as effectively achieved through watching.",
            "Reading also promotes better concentration and focus, as it requires active participation from the reader.",
            "Finally, reading is a more personal experience, allowing individuals to interpret and visualize the story in their own unique way."
        ]

        self.reading_text = """Reading is a more engaging activity compared to watching.
                    It stimulates the imagination and enhances cognitive functions in ways that watching cannot.
                    Books often provide a deeper understanding of characters and plot, allowing for a more immersive experience.
                    Furthermore, reading improves vocabulary and language skills, which is not as effectively achieved through watching.
                    Reading also promotes better concentration and focus, as it requires active participation from the reader.
                    Finally, reading is a more personal experience, allowing individuals to interpret and visualize the story in their own unique way."""

    def calculate_coherence(self, segment):
        embeddings = self.embedding_model.encode(segment)
        cosine_similarities = np.inner(embeddings, embeddings)
        avg_similarity = np.mean(cosine_similarities)
        return avg_similarity

    def refine_segments(self, segments, min_coherence=0.6):
        refined_segments = []
        for segment in segments:
            if self.calculate_coherence(segment) < min_coherence:
                # If coherence is low, split the segment
                split_index = len(segment) // 2
                refined_segments.append(segment[:split_index])
                refined_segments.append(segment[split_index:])
            else:
                refined_segments.append(segment)
        return refined_segments

    def test_chess_sentences(self):
        print("\nTesting with Chess sentences:\n")
        self.run_segmentation_test(self.chess_sentences, "chess")

    def test_chess_text_block(self):
        print("\nTesting with Chess text block:\n")
        # Split the text block into sentences
        argument = sent_tokenize(self.chess_text)
        self.run_segmentation_test(argument, "chess")

    def test_reading_sentences(self):
        print("\nTesting with Reading sentences:\n")
        self.run_segmentation_test(self.reading_sentences, "reading")

    def test_reading_text_block(self):
        print("\nTesting with Reading text block:\n")
        # Split the text block into sentences
        argument = sent_tokenize(self.reading_text)
        self.run_segmentation_test(argument, "reading")

    def run_segmentation_test(self, argument, example_type):
        # Step 2: Create Sentence Pairs
        sentence_pairs = []
        for i in range(len(argument) - 1):
            sentence_pairs.append([argument[i], argument[i + 1]])

        # Assert sentence pairs are correctly formed
        self.assertEqual(len(sentence_pairs), len(argument) - 1)
        print("Sentence pairs correctly formed.")

        # Step 3: Predict Segmentation Scores
        scores = self.model.predict(sentence_pairs)

        # Assert scores are calculated for each pair
        self.assertEqual(len(scores), len(sentence_pairs))
        print("Segmentation scores calculated for each pair.")

        # Compute distance/differentiation scores
        differentiation_scores = [abs(score[2] - score[0]) for score in scores]

        # Print differentiation scores
        print("\nDifferentiation Scores:")
        for i, diff_score in enumerate(differentiation_scores):
            print(f"Differentiation score for pair {i}: {diff_score}")

        # Assert differentiation scores are calculated for each pair
        self.assertEqual(len(differentiation_scores), len(scores))
        print("Differentiation scores calculated for each pair.")

        # Calculate the trimmed mean, trimming 10% of the smallest and largest values
        threshold = trim_mean(differentiation_scores, 0.1)

        print(f"\nThreshold: {threshold}")

        # Step 4: Interpret Scores and Identify Boundaries
        segments = []
        current_segment = [argument[0]]

        for i in range(1, len(argument)):
            if differentiation_scores[i - 1] > threshold:
                # If the differentiation score is above the threshold, start a new segment
                segments.append(current_segment)
                current_segment = [argument[i]]
                print(f"\nNew segment started at sentence {i}")
            else:
                # If the differentiation score is below the threshold, continue the current segment
                current_segment.append(argument[i])
                print(f"Continuing current segment at sentence {i}")

        # Add the last segment
        if current_segment:
            segments.append(current_segment)

        # Print final segments
        print("\nFinal Segments:")
        for i, segment in enumerate(segments):
            print(f"Segment {i}: {segment}")

        # Refinement step based on coherence
        refined_segments = self.refine_segments(segments)
        print("\nRefined Segments:")
        for i, segment in enumerate(refined_segments):
            print(f"Segment {i}: {segment}")

        # Expected segmentation result based on the example argument
        expected_segments = {
            "chess": [
                ["Chess is a game of deeper strategy compared to checkers."],
                [
                    "It offers a complexity that requires players to think several moves ahead, promoting strategic thinking and planning skills.",
                    "Each piece in chess has its own unique moves and capabilities, unlike the uniform pieces in checkers, adding layers of strategy and tactics.",
                    "Furthermore, chess has a rich history and cultural significance that checkers lacks."
                ],
                [
                    "The game has been played by kings and commoners alike for centuries and has influenced various aspects of art, literature, and even politics."
                ],
                ["This cultural depth adds to the enjoyment and appreciation of the game.",
                 "Chess also offers more varied and challenging gameplay."],
                [
                    "The opening moves alone in chess provide a nearly infinite number of possibilities, leading to different game progressions each time.",
                    "Checkers, by contrast, has a more limited set of opening moves, which can make the game feel repetitive over time.",
                    "Finally, chess is recognized globally as a competitive sport with international tournaments and rankings."
                ],
                [
                    "This global recognition and the opportunities for competition at all levels make chess a more engaging and rewarding game for those who enjoy not only playing but also watching and studying the game."
                ]
            ],
            "reading": [
                ["Reading is a more engaging activity compared to watching."],
                ["It stimulates the imagination and enhances cognitive functions in ways that watching cannot."],
                [
                    "Books often provide a deeper understanding of characters and plot, allowing for a more immersive experience.",
                    "Furthermore, reading improves vocabulary and language skills, which is not as effectively achieved through watching."
                ],
                [
                    "Reading also promotes better concentration and focus, as it requires active participation from the reader.",
                    "Finally, reading is a more personal experience, allowing individuals to interpret and visualize the story in their own unique way."
                ]
            ]
        }

        if "chess" in argument[0].lower():
            expected = expected_segments["chess"]
        else:
            expected = expected_segments["reading"]

        # Verify the segmentation result matches the expected result
        self.assertEqual(segments, expected)
        print("Segments match the expected output.")

if __name__ == '__main__':
    unittest.main()
