import unittest
from sentence_transformers import CrossEncoder
import sentencepiece  # Ensure sentencepiece is installed
import google.protobuf  # Ensure protobuf is installed
from nltk.tokenize import sent_tokenize

class TestArgumentSegmentation(unittest.TestCase):

    def setUp(self):
        # Initialize the Cross-Encoder model
        self.model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

        # Example long argument for the affirmative side of the topic "Chess is better than checkers"
        self.argument_sentences = [
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

        self.argument_text = """Chess is a game of deeper strategy compared to checkers.
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

    def test_segmentation_sentences(self):
        print("\nTesting with individual sentences:\n")
        self.run_segmentation_test(self.argument_sentences)

    def test_segmentation_text_block(self):
        print("\nTesting with text block:\n")
        # Split the text block into sentences
        argument = sent_tokenize(self.argument_text)
        self.run_segmentation_test(argument)

    def run_segmentation_test(self, argument):
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

        # Determine threshold dynamically if needed
        threshold = sum(differentiation_scores) / len(differentiation_scores)  # Example: average differentiation score

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

        # Expected segmentation result based on the example argument
        expected_segments = [
            ["Chess is a game of deeper strategy compared to checkers."],
            [
                "It offers a complexity that requires players to think several moves ahead, promoting strategic thinking and planning skills.",
                "Each piece in chess has its own unique moves and capabilities, unlike the uniform pieces in checkers, adding layers of strategy and tactics.",
                "Furthermore, chess has a rich history and cultural significance that checkers lacks."
            ],
            ["The game has been played by kings and commoners alike for centuries and has influenced various aspects of art, literature, and even politics."],
            ["This cultural depth adds to the enjoyment and appreciation of the game.", "Chess also offers more varied and challenging gameplay."],
            [
                "The opening moves alone in chess provide a nearly infinite number of possibilities, leading to different game progressions each time.",
                "Checkers, by contrast, has a more limited set of opening moves, which can make the game feel repetitive over time.",
                "Finally, chess is recognized globally as a competitive sport with international tournaments and rankings."
            ],
            ["This global recognition and the opportunities for competition at all levels make chess a more engaging and rewarding game for those who enjoy not only playing but also watching and studying the game."]
        ]

        # Verify the segmentation result matches the expected result
        self.assertEqual(segments, expected_segments)
        print("Segments match the expected output.")

if __name__ == '__main__':
    unittest.main()
