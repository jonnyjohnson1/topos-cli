import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np

# Ensure the necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Example sentences
sentences = [
    "Chess is a game of deeper strategy compared to checkers.",
    "It offers a complexity that requires players to think several moves ahead.",
    "Chess has a rich history.",
    "Reading stimulates the imagination.",
    "Books provide a deeper understanding of characters and plot."
]

# Step 1: Obtain Sentence Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

# Step 2: Identify Hypernyms
def get_hypernyms(sentence):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    hypernyms = []
    for word, pos in pos_tags:
        synsets = wn.synsets(word, pos=nltk.corpus.reader.wordnet.NOUN)
        if synsets:
            hypernym = synsets[0].hypernyms()
            if hypernym:
                hypernyms.append(hypernym[0].lemma_names()[0])
    return hypernyms

# Identify hypernyms for each sentence
sentence_hypernyms = [get_hypernyms(sentence) for sentence in sentences]

# Step 3: Calculate Spin Factor
def calculate_spin_factor(hypernyms, target_hypernym):
    return sum(1 for hypernym in hypernyms if hypernym == target_hypernym)

# Example target hypernym
target_hypernym = "game"

# Calculate spin factor for each sentence
spin_factors = [calculate_spin_factor(hypernyms, target_hypernym) for hypernyms in sentence_hypernyms]

# Step 4: Adjust Similarity Scores
# Calculate initial similarity scores (cosine similarity)
initial_similarity = [[1 - cosine(e1, e2) for e2 in embeddings] for e1 in embeddings]

# Adjust similarity scores using spin factors
adjusted_similarity = initial_similarity.copy()
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if spin_factors[i] == spin_factors[j]:
            adjusted_similarity[i][j] += 0.1  # Increase similarity if spin factors are aligned

# Step 5: Perform Hierarchical Clustering
# Convert adjusted similarity to distance
distance_matrix = 1 - np.array(adjusted_similarity)
Z = linkage(distance_matrix, method='ward')

# Step 6: Plot the Dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=[f'Sentence {i+1}' for i in range(len(sentences))])
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sentences')
plt.ylabel('Distance')
plt.show()
