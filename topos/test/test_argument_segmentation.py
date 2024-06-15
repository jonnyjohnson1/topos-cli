import unittest
from sentence_transformers import CrossEncoder
import sentencepiece  # Ensure sentencepiece is installed
import google.protobuf  # Ensure protobuf is installed


class TestArgumentSegmentation(unittest.TestCase):

    def setUp(self):
        # Initialize the Cross-Encoder model
        self.model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

        # Example long argument for the affirmative side of the topic "Chess is better than checkers"
        self.argument = [
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

    def test_segmentation(self):
        # Step 1: Tokenization and Sentence Segmentation is already provided in the setUp as a list of sentences

        # Step 2: Create Sentence Pairs
        sentence_pairs = []
        for i in range(len(self.argument) - 1):
            sentence_pairs.append([self.argument[i], self.argument[i + 1]])

        # Print sentence pairs
        print("Sentence Pairs:")
        for pair in sentence_pairs:
            print(pair)

        # Step 3: Predict Segmentation Scores
        scores = self.model.predict(sentence_pairs)

        # Print segmentation scores
        print("\nSegmentation Scores:")
        for i, score in enumerate(scores):
            print(f"Score for pair {i}: {score}")

        # Set a threshold for segmentation
        threshold = 0.5  # This value can be adjusted based on the model's output

        # Step 4: Interpret Scores and Identify Boundaries
        segments = []
        current_segment = [self.argument[0]]

        for i in range(1, len(self.argument)):
            max_score = max(scores[i - 1])
            if max_score < threshold:
                # If the maximum score is below the threshold, start a new segment
                segments.append(current_segment)
                current_segment = [self.argument[i]]
                print(f"\nNew segment started at sentence {i} with max score {max_score}")
            else:
                # If the maximum score is above the threshold, continue the current segment
                current_segment.append(self.argument[i])
                print(f"Continuing current segment at sentence {i} with max score {max_score}")

        # Add the last segment
        if current_segment:
            segments.append(current_segment)

        # Print final segments
        print("\nFinal Segments:")
        for i, segment in enumerate(segments):
            print(f"Segment {i}: {segment}")

        # Expected segmentation result based on the example argument
        expected_segments = [
            [
                "Chess is a game of deeper strategy compared to checkers.",
                "It offers a complexity that requires players to think several moves ahead, promoting strategic thinking and planning skills.",
                "Each piece in chess has its own unique moves and capabilities, unlike the uniform pieces in checkers, adding layers of strategy and tactics."
            ],
            [
                "Furthermore, chess has a rich history and cultural significance that checkers lacks.",
                "The game has been played by kings and commoners alike for centuries and has influenced various aspects of art, literature, and even politics.",
                "This cultural depth adds to the enjoyment and appreciation of the game."
            ],
            [
                "Chess also offers more varied and challenging gameplay.",
                "The opening moves alone in chess provide a nearly infinite number of possibilities, leading to different game progressions each time.",
                "Checkers, by contrast, has a more limited set of opening moves, which can make the game feel repetitive over time."
            ],
            [
                "Finally, chess is recognized globally as a competitive sport with international tournaments and rankings.",
                "This global recognition and the opportunities for competition at all levels make chess a more engaging and rewarding game for those who enjoy not only playing but also watching and studying the game."
            ]
        ]

        # Verify the segmentation result matches the expected result
        self.assertEqual(segments, expected_segments)


if __name__ == '__main__':
    unittest.main()
