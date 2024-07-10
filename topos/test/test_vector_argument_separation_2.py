import umap
import unittest
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from bertopic import BERTopic
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Input data
message_data = [
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Human activity, particularly the burning of fossil fuels, has significantly increased the concentration of greenhouse gases in the atmosphere."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Climate change is a natural phenomenon that has occurred throughout Earth's history, long before human activity."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Scientific studies show a direct correlation between industrialization and the rise in global temperatures."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "There are many natural factors, such as volcanic eruptions and solar radiation, that contribute to global temperature changes."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "The Intergovernmental Panel on Climate Change (IPCC) reports that human influence is the dominant cause of global warming since the mid-20th century."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "The climate models used to predict human impact are based on assumptions that might not accurately reflect complex climate systems."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Deforestation for agriculture and urban development reduces the Earth's capacity to absorb CO2, exacerbating climate change."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "The Earth's climate has always fluctuated; there were periods of warming and cooling even in the pre-industrial era."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Human-driven emissions of CO2 and methane are accelerating the melting of polar ice caps and glaciers."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Some studies suggest that current climate changes could be part of a natural cycle, not necessarily driven by human activities."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "While natural factors do play a role, the rapid increase in CO2 levels coincides with the industrial revolution and increased fossil fuel use."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Historical data shows that the Earth has experienced higher CO2 levels and warmer temperatures in the past without human influence."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "The majority of climate scientists agree that human activity is the primary driver of recent climate change."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Some scientists argue that the impact of human activities on climate change is overstated."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "The acidification of oceans due to increased CO2 levels is a direct result of human emissions, affecting marine ecosystems."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "The costs of transitioning to renewable energy are high and could negatively impact economies."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Extreme weather events, such as hurricanes and wildfires, have become more frequent and severe due to human-induced climate change."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Efforts to combat climate change should focus on adaptation rather than trying to control the climate."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Renewable energy sources like wind and solar power can reduce our dependence on fossil fuels and mitigate climate change."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Developing countries need affordable energy to grow their economies, and restricting fossil fuel use could hinder their development."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Ignoring the human impact on climate change could lead to catastrophic consequences for future generations."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "It's important to consider the economic implications of drastic measures to reduce emissions."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Investing in green technology creates jobs and boosts the economy while protecting the environment."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Developing new technologies for adaptation can also create jobs and economic growth."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "International agreements like the Paris Accord aim to reduce global emissions and limit temperature rise, benefiting everyone."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Policies should balance environmental concerns with economic needs to avoid undue hardship."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "The health impacts of air pollution from fossil fuels are significant, causing respiratory problems and premature deaths."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Natural climate variability must be accounted for in any climate policy."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Preserving natural habitats and biodiversity is crucial for a sustainable future and mitigating climate change."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Focusing on resilience and adaptation strategies can help communities better cope with climate changes."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Proactive measures to reduce emissions will ultimately be more cost-effective than dealing with the aftermath of climate disasters."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "There's still uncertainty in the extent of human impact on climate change."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Global cooperation is essential to address a problem that affects all nations."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Climate policies should be flexible and adaptable to new scientific findings."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "The transition to a green economy is an opportunity to innovate and lead in sustainable practices."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Economic growth and environmental protection can go hand in hand with smart policies."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Public awareness and education about climate change can drive more sustainable behaviors."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Innovation in carbon capture and storage technologies can reduce emissions without drastic lifestyle changes."}},
    {"role": "user", "data": {"user_id": "userA",
                              "content": "Protecting the environment is a moral responsibility to ensure a livable planet for future generations."}},
    {"role": "user", "data": {"user_id": "userB",
                              "content": "Addressing climate change requires a balanced approach that considers both environmental and economic factors."}}
]


# Set display options to show the full DataFrame without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 80)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 80)


class TestBERTopicSVM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sentences = [msg["data"]["content"] for msg in message_data]
        cls.users = [msg["data"]["user_id"] for msg in message_data]
        cls.user_colors = {'userA': 'blue', 'userB': 'green'}
        cls.user_data = {user: [] for user in set(cls.users)}

        # Distribute messages to each user
        for i, sentence in enumerate(cls.sentences):
            cls.user_data[cls.users[i]].append(sentence)

    def process_user_messages(self, user_id):
        user_messages = self.user_data[user_id]
        print(f"Processing messages for {user_id}:\n")
        print(user_messages)

        # Initialize variables for incremental processing
        processed_messages = []
        all_topics = []

        for i, message in enumerate(user_messages):
            processed_messages.append(message)

            # Proceed only if there are at least two messages
            if len(processed_messages) < 2:
                continue

            # Apply BERTopic incrementally
            try:
                if len(processed_messages) < 5:
                    # Use a smaller n_neighbors for UMAP when the dataset is small
                    topic_model = BERTopic(
                        umap_model=umap.UMAP(n_neighbors=len(processed_messages) - 1, min_dist=0.1, n_components=5,
                                             random_state=42))
                else:
                    topic_model = BERTopic()
                topics, probs = topic_model.fit_transform(processed_messages)
            except ValueError as e:
                print(f"Skipping step {i + 1} due to insufficient data for UMAP: {e}")
                continue
            all_topics.append(topics[-1])  # Only add the latest topic

            # Get the topic information
            topic_info = topic_model.get_topic_info()
            print("Topic Info:")
            print(topic_info)

            # Print topics and associated messages
            for topic in topic_info.Topic:
                if len(topic_info.Topic) == 1:
                    if topic == -1:
                        continue
                topic_name = topic_info[topic_info.Topic == topic]['Name'].values[0]
                print(f"\nTopic {topic} - {topic_name}:")
                topic_docs = [processed_messages[i] for i, t in enumerate(topics) if t == topic]
                for doc in topic_docs:
                    print(f" - {doc}")

            # Visualize topics incrementally
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = range(len(processed_messages))
            colors = [self.user_colors[user_id]] * len(processed_messages)
            scatter = ax.scatter(indices, topics, color=colors)

            ax.legend()
            plt.xlabel('Sentence Index')
            plt.ylabel('Topic')
            plt.title(f'Topic Distribution for {user_id} at step {i + 1}')
            plt.show()

            # Train SVM classifier incrementally
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(processed_messages)
            y = pd.Series(topics)

            # Split data into training and test sets with stratification
            if len(y.unique()) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

                # Create an SVM pipeline with parameter tuning
                svm_pipeline = make_pipeline(StandardScaler(with_mean=False),
                                             SVC(kernel='linear', probability=True, C=0.55))
                svm_pipeline.fit(X_train, y_train)

                # Predict and evaluate incrementally
                y_pred = svm_pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f'SVM Accuracy for {user_id} at step {i + 1}: {accuracy * 100:.2f}%')

                if i > 0:  # Only check accuracy after the first step
                    self.assertGreater(accuracy, 0.7, f"SVM accuracy for {user_id} should be above 70% at step {i + 1}")

                # Visualize SVM predictions incrementally
                svm_predictions = svm_pipeline.predict(X)
                df = pd.DataFrame({'Sentence': processed_messages, 'SVM_Prediction': svm_predictions})

                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(indices, svm_predictions, color=colors)

                plt.xlabel('Sentence Index')
                plt.ylabel('SVM Predicted Topic')
                plt.title(f'SVM Predicted Topic Distribution for {user_id} at step {i + 1}')
                plt.show()
            else:
                print(f"Not enough classes in the data to train SVM for {user_id} at step {i + 1}.")

            # Print clustered sentences by topic incrementally
            for topic in y.unique():
                if topic == -1:
                    continue
                topic_name = topic_info[topic_info.Topic == topic]['Name'].values[0]
                print(f"\nTopic {topic} - {topic_name}:")
                topic_docs = [processed_messages[i] for i, t in enumerate(topics) if t == topic]
                for doc in topic_docs:
                    print(f" - {doc}")

    def test_process_users(self):
        for user_id in self.user_data:
            self.process_user_messages(user_id)

if __name__ == '__main__':
    unittest.main()
