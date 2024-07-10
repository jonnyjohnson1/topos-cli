import numpy as np
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
from nltk.corpus import wordnet
# import gensim.downloader as api

class HumorCalculator:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        # self.word_vectors = api.load("glove-wiki-gigaword-100")
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')

    def calculate_humor_score(self, target_sentence, context_before, context_after):
        context = context_before + [target_sentence] + context_after
        features = {}

        features['sentiment_contrast'] = self.sentiment_contrast(target_sentence, context)
        features['semantic_incongruity'] = self.semantic_incongruity(target_sentence, context)
        features['sarcasm_score'] = self.detect_sarcasm(target_sentence, context)
        features['lexical_ambiguity'] = self.lexical_ambiguity(target_sentence)
        features['unexpectedness'] = self.calculate_unexpectedness(target_sentence, context)
        features['emotional_arousal'] = self.emotional_arousal(target_sentence)
        features['formality_shift'] = self.formality_shift(target_sentence, context)
        features['cultural_reference'] = self.cultural_reference_check(target_sentence)
        features['rhetorical_devices'] = self.identify_rhetorical_devices(target_sentence)
        features['timing_rhythm'] = self.analyze_timing_rhythm(target_sentence)
        features['self_reference'] = self.detect_self_reference(target_sentence)
        features['contextual_contrast'] = self.contextual_contrast(target_sentence, context)
        features['subverted_expectations'] = self.subverted_expectations(target_sentence, context_before)
        features['repetition_callback'] = self.detect_repetition_callback(target_sentence, context)

        # Combine features into a single humor score
        humor_score = sum(features.values()) / len(features)
        return humor_score, features

    def sentiment_contrast(self, sentence, context):
        target_sentiment = self.sentiment_analyzer(sentence)[0]['score']
        context_sentiment = np.mean([self.sentiment_analyzer(s)[0]['score'] for s in context])
        return abs(target_sentiment - context_sentiment)

    def semantic_incongruity(self, sentence, context):
        sentence_embedding = self.sentence_model.encode(sentence)
        context_embedding = self.sentence_model.encode(" ".join(context))
        return 1 - np.dot(sentence_embedding, context_embedding) / (np.linalg.norm(sentence_embedding) * np.linalg.norm(context_embedding))

    def detect_sarcasm(self, sentence, context):
        sentiment = TextBlob(sentence).sentiment.polarity
        content_words = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(sentence)) if pos.startswith('JJ') or pos.startswith('RB')]
        content_sentiment = np.mean([TextBlob(word).sentiment.polarity for word in content_words]) if content_words else 0
        return abs(sentiment - content_sentiment)

    def lexical_ambiguity(self, sentence):
        words = nltk.word_tokenize(sentence)
        ambiguity_scores = [len(wordnet.synsets(word)) for word in words]
        return np.mean(ambiguity_scores) if ambiguity_scores else 0

    def calculate_unexpectedness(self, sentence, context):
        context_words = set(" ".join(context).split())
        sentence_words = set(sentence.split())
        unexpected_words = sentence_words - context_words
        return len(unexpected_words) / len(sentence_words) if sentence_words else 0

    def emotional_arousal(self, sentence):
        emotions = self.emotion_classifier(sentence)[0]
        return max(emotion['score'] for emotion in emotions)

    def formality_shift(self, sentence, context):
        formal_words = set(["moreover", "furthermore", "consequently", "thus", "hence", "ergo"])
        informal_words = set(["yeah", "nah", "gonna", "wanna", "sorta", "kinda"])
        sentence_formality = len([word for word in sentence.split() if word.lower() in formal_words]) - len([word for word in sentence.split() if word.lower() in informal_words])
        context_formality = np.mean([len([word for word in s.split() if word.lower() in formal_words]) - len([word for word in s.split() if word.lower() in informal_words]) for s in context])
        return abs(sentence_formality - context_formality)

    def cultural_reference_check(self, sentence):
        # This would require a comprehensive database of cultural references
        # For simplicity, we'll check for a few example references
        references = ["rickroll", "game of thrones", "star wars", "brexit", "covfefe"]
        return any(ref in sentence.lower() for ref in references)

    def identify_rhetorical_devices(self, sentence):
        devices = {
            'hyperbole': ["always", "never", "everyone", "no one", "best", "worst"],
            'understatement': ["slightly", "a bit", "somewhat", "rather"],
            'irony': ["great", "fantastic", "wonderful", "brilliant"]  # in negative contexts
        }
        return sum(word in sentence.lower() for device in devices.values() for word in device) / len(sentence.split())

    def analyze_timing_rhythm(self, sentence):
        words = sentence.split()
        return 1 - (abs(len(words) - 10) / 10)  # Assumes optimal length around 10 words

    def detect_self_reference(self, sentence):
        self_references = ["I", "me", "my", "myself", "we", "us", "our"]
        return any(word in sentence.lower().split() for word in self_references)

    def contextual_contrast(self, sentence, context):
        sentence_embedding = self.sentence_model.encode(sentence)
        context_embedding = self.sentence_model.encode(" ".join(context))
        return 1 - np.dot(sentence_embedding, context_embedding) / (np.linalg.norm(sentence_embedding) * np.linalg.norm(context_embedding))

    def subverted_expectations(self, sentence, context_before):
        # Simple implementation: check if sentence starts with "But" or "However"
        return int(sentence.lower().startswith(("but", "however")))

    def detect_repetition_callback(self, sentence, context):
        words = sentence.split()
        context_words = " ".join(context).split()
        repetitions = sum(words.count(word) > 1 for word in set(words))
        callbacks = sum(word in context_words for word in words)
        return (repetitions + callbacks) / len(words)

# Usage
calculator = HumorCalculator()
target_sentence = "I'm not saying it's aliens... but it's aliens."
context_before = ["We've been studying these mysterious crop circles for years.", "Scientists are baffled by their complexity and precision."]
context_after = ["Of course, that's just my professional opinion as a conspiracy theorist.", "Don't quote me on that in your academic papers."]

humor_score, features = calculator.calculate_humor_score(target_sentence, context_before, context_after)
print(f"Humor Score: {humor_score}")
print("Feature Breakdown:")
for feature, value in features.items():
    print(f"  {feature}: {value}")