import re

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Download required NLTK data
def download_nltk_data():
    resources = ["punkt_tab", "stopwords"]
    for resource in resources:
        try:
            nltk.data.find(
                f"tokenizers/{resource}"
                if resource == "punkt_tab"
                else f"corpora/{resource}"
            )
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)


# Download NLTK data at module import
download_nltk_data()


class ClinicalNoteClassifier:
    def __init__(self, model_type="logistic"):
        """
        Initialize the classifier
        Args:
            model_type (str): Either 'logistic' or 'random_forest'
        """
        self.model_type = model_type
        self.pipeline = None
        self.label_encoder = None

    def preprocess_text(self, text):
        """
        Preprocess the text by:
        1. Converting to lowercase
        2. Removing special characters and numbers
        3. Removing extra whitespace
        4. Removing stopwords
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text)
        text = " ".join([word for word in word_tokens if word not in stop_words])

        return text

    def create_pipeline(self):
        """
        Create the ML pipeline with TF-IDF vectorizer and classifier
        """
        if self.model_type == "logistic":
            classifier = LogisticRegression(
                max_iter=1000, C=1.0, class_weight="balanced", random_state=42
            )
        else:  # random_forest
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
            )

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=50000, ngram_range=(1, 2), min_df=2, max_df=0.95
                    ),
                ),
                ("classifier", classifier),
            ]
        )

    def train(self, texts, labels):
        """
        Train the model
        Args:
            texts (list): List of clinical notes
            labels (list): List of corresponding labels
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Create and train pipeline
        self.create_pipeline()
        self.pipeline.fit(processed_texts, labels)

        # Evaluate on training set
        predictions = self.pipeline.predict(processed_texts)
        print("\nTraining Set Performance:")
        print(classification_report(labels, predictions))

    def predict(self, text):
        """
        Make prediction for a single text
        Args:
            text (str): Clinical note text
        Returns:
            dict: Dictionary containing top 3 predictions with probabilities
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Get prediction probabilities
        probabilities = self.pipeline.predict_proba([processed_text])[0]

        # Get top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        top3_probs = probabilities[top3_indices]

        predictions = []
        for prob, idx in zip(top3_probs, top3_indices):
            predictions.append(
                {"label": self.pipeline.classes_[idx], "probability": float(prob)}
            )

        return predictions

    def save_model(self, path):
        """
        Save the trained model
        Args:
            path (str): Path to save the model
        """
        joblib.dump(self.pipeline, path)

    def load_model(self, path):
        """
        Load a trained model
        Args:
            path (str): Path to the saved model
        """
        self.pipeline = joblib.load(path)


# Example usage:
if __name__ == "__main__":
    # Example data
    texts = [
        "Patient presents with severe headache and fever",
        "Routine checkup completed, all vitals normal",
        "Follow-up appointment for diabetes management",
    ]
    labels = ["Neurology", "General Medicine", "Endocrinology"]

    # Create and train classifier
    classifier = ClinicalNoteClassifier(model_type="logistic")
    classifier.train(texts, labels)

    # Make prediction
    test_text = "Patient reports persistent headache and dizziness"
    predictions = classifier.predict(test_text)
    print("\nPredictions for test text:")
    for pred in predictions:
        print(f"{pred['label']}: {pred['probability']:.2%}")
