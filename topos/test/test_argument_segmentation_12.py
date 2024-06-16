import unittest
from topos.FC.argument_detection import ArgumentDetection


class TestArgumentSegmentation(unittest.TestCase):

    def setUp(self):
        self.argument_detection = ArgumentDetection(api_key="your_api_key")

        # Example sentences for the debate topic
        self.debate_sentences = [
            "Social media platforms have become the primary source of information for many people.",
            "They have the power to influence public opinion and election outcomes.",
            "Government regulation could help in mitigating the spread of false information.",
            "On the other hand, government intervention might infringe on freedom of speech.",
            "Social media companies are already taking steps to address misinformation.",
            "Self-regulation is preferable as it avoids the risks of government overreach.",
            "The lack of regulation has led to the proliferation of harmful content and echo chambers."
        ]

        # Example sentences for Chess vs. Checkers
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

        # Example sentences for Reading vs. Watching
        self.reading_sentences = [
            "Reading is a more engaging activity compared to watching.",
            "It stimulates the imagination and enhances cognitive functions in ways that watching cannot.",
            "Books often provide a deeper understanding of characters and plot, allowing for a more immersive experience.",
            "Furthermore, reading improves vocabulary and language skills, which is not as effectively achieved through watching.",
            "Reading also promotes better concentration and focus, as it requires active participation from the reader.",
            "Finally, reading is a more personal experience, allowing individuals to interpret and visualize the story in their own unique way."
        ]

    def test_argument_segmentation(self):
        debate_clusters = self.argument_detection.cluster_sentences(self.debate_sentences)
        chess_clusters = self.argument_detection.cluster_sentences(self.chess_sentences)
        reading_clusters = self.argument_detection.cluster_sentences(self.reading_sentences)

        # Expected segmentation result based on the debate arguments
        expected_segments = {
            "debate": [
                [
                    "Social media platforms have become the primary source of information for many people.",
                    "They have the power to influence public opinion and election outcomes."
                ],
                [
                    "Government regulation could help in mitigating the spread of false information.",
                    "The lack of regulation has led to the proliferation of harmful content and echo chambers."
                ],
                ["On the other hand, government intervention might infringe on freedom of speech."],
                [
                    "Social media companies are already taking steps to address misinformation.",
                    "Self-regulation is preferable as it avoids the risks of government overreach."
                ]
            ],
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

        print("\n[INFO] Actual Debate Segments:")
        for segment in debate_clusters.values():
            print(segment)

        print("\n[INFO] Expected Debate Segments:")
        for segment in expected_segments["debate"]:
            print(segment)

            print("\n[INFO] Actual Chess Segments:")
            for segment in chess_clusters.values():
                print(segment)

            print("\n[INFO] Expected Chess Segments:")
            for segment in expected_segments["chess"]:
                print(segment)

            print("\n[INFO] Actual Reading Segments:")
            for segment in reading_clusters.values():
                print(segment)

            print("\n[INFO] Expected Reading Segments:")
            for segment in expected_segments["reading"]:
                print(segment)

            # Assertions to check if the actual segmentation matches the expected segmentation
            actual_debate_segments = list(debate_clusters.values())
            actual_chess_segments = list(chess_clusters.values())
            actual_reading_segments = list(reading_clusters.values())

            self.assertEqual(actual_debate_segments, expected_segments["debate"])
            self.assertEqual(actual_chess_segments, expected_segments["chess"])
            self.assertEqual(actual_reading_segments, expected_segments["reading"])

    if __name__ == '__main__':
        unittest.main()
