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


class TestBERTopicSVM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sentences = [msg["data"]["content"] for msg in message_data]
        cls.users = [msg["data"]["user_id"] for msg in message_data]
        cls.user_colors = {'userA': 'blue', 'userB': 'green'}

        # Apply BERTopic
        cls.topic_model = BERTopic()
        cls.topics, _ = cls.topic_model.fit_transform(cls.sentences)

        # Get topic representations
        cls.topic_info = cls.topic_model.get_topic_info()
        print(cls.topic_info)

        # Get the document info to extract topic labels
        cls.document_info = cls.topic_model.get_document_info(cls.sentences)
        cls.topic_labels = cls.document_info['Topic'].values

        # Create a DataFrame for visualization
        cls.df = pd.DataFrame({'Sentence': cls.sentences, 'User': cls.users, 'Topic': cls.topic_labels})

    def test_visualize_topics(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [self.user_colors[user] for user in self.df['User']]
        scatter = ax.scatter(self.df.index, self.df['Topic'], c=colors)

        # Create a legend
        legend_labels = {v: k for k, v in self.user_colors.items()}
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mfc, markersize=10) for mfc in self.user_colors.values()]
        ax.legend(handles, legend_labels.values())

        plt.xlabel('Sentence Index')
        plt.ylabel('Topic')
        plt.title('Topic Distribution per Sentence Colored by User')
        plt.show()

    def test_train_svm(self):
        # Train SVM classifier
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.sentences)
        y = self.df['Topic']

        # Split data into training and test sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Check if we have more than one class in the training set
        if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
            # Create an SVM pipeline
            svm_pipeline = make_pipeline(StandardScaler(with_mean=False), SVC(kernel='linear', probability=True))
            svm_pipeline.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = svm_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'SVM Accuracy: {accuracy * 100:.2f}%')

            self.assertGreater(accuracy, 0.7, "SVM accuracy should be above 70%")

            # Visualize SVM predictions
            svm_predictions = svm_pipeline.predict(X)
            self.df['SVM_Prediction'] = svm_predictions

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = [self.user_colors[user] for user in self.df['User']]
            scatter = ax.scatter(self.df.index, self.df['SVM_Prediction'], c=colors)

            # Create a legend
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mfc, markersize=10) for mfc in self.user_colors.values()]
            ax.legend(handles, legend_labels.values())

            plt.xlabel('Sentence Index')
            plt.ylabel('SVM Predicted Topic')
            plt.title('SVM Predicted Topic Distribution per Sentence Colored by User')
            plt.show()
        else:
            print("Not enough classes in the training or test set to train SVM.")

    def test_print_clustered_sentences(self):
        for topic in self.df['Topic'].unique():
            print(f"Topic {topic}:")
            topic_sentences = self.df[self.df['Topic'] == topic]['Sentence'].tolist()
            for sentence in topic_sentences:
                print(f" - {sentence}")
            print()

if __name__ == '__main__':
    unittest.main()
