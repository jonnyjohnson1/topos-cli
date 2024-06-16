from sentence_transformers import CrossEncoder
import matplotlib.pyplot as plt
import numpy as np


# Function to calculate similarity scores for a given model
def calculate_similarity(model, sentences):
    scores = []
    for i in range(len(sentences) - 2):
        pair1 = (sentences[i], sentences[i + 1])
        pair2 = (sentences[i + 1], sentences[i + 2])
        score1 = model.predict([pair1])[0]
        score2 = model.predict([pair2])[0]
        scores.append((score1, score2))
    return scores


# Function to plot similarity scores for multiple models
def plot_similarity_scores(models, model_names, sentences):
    fig, axs = plt.subplots(len(models), 3, figsize=(15, 5 * len(models)))
    score_labels = ['Contradiction', 'Entailment', 'Neutral']

    for m, (model, model_name) in enumerate(zip(models, model_names)):
        similarity_scores = calculate_similarity(model, sentences)
        for i in range(3):
            axs[m, i].plot([score[i] for pair in similarity_scores for score in pair], marker='o')
            axs[m, i].set_title(f'{model_name} - {score_labels[i]}')
            axs[m, i].set_xlabel('Sentence Pair Index')
            axs[m, i].set_ylabel('Score')

    plt.tight_layout()
    plt.show()


# Load models
models = [
    CrossEncoder('cross-encoder/nli-deberta-v3-base'),
    CrossEncoder('cross-encoder/nli-deberta-v3-large'),
    CrossEncoder('cross-encoder/nli-deberta-v3-small'),
    CrossEncoder('cross-encoder/nli-deberta-v3-xsmall'),
]
model_names = ['nli-deberta-v3-base', 'nli-deberta-v3-large', 'nli-deberta-v3-small', 'nli-deberta-v3-xsmall']

# Example sentences
sentences = [
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

# Run the visualization
plot_similarity_scores(models, model_names, sentences)
