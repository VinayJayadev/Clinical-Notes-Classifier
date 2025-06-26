import os

import joblib
import pandas as pd

from classical_ml_classifier import ClinicalNoteClassifier


def train_model(data_path, model_type="logistic"):
    """
    Train the enhanced classical ML model with keywords
    Args:
        data_path (str): Path to the training data CSV file
        model_type (str): Either 'logistic' or 'random_forest'
    """
    # Create results directory if it doesn't exist
    os.makedirs("./results/classical_ml_enhanced", exist_ok=True)

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Verify required columns exist
    required_columns = ["transcription", "medical_specialty"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")

    # Remove rows with missing values in required columns
    df = df.dropna(subset=required_columns)

    # Enhance text with keywords if available
    print("Enhancing text with keywords...")
    enhanced_texts = []
    for idx, row in df.iterrows():
        text = row["transcription"]

        # Add keywords if available
        if "keywords" in df.columns and pd.notna(row["keywords"]):
            keywords = str(row["keywords"])
            # Combine transcription with keywords
            enhanced_text = f"{text} {keywords}"
        else:
            enhanced_text = text

        enhanced_texts.append(enhanced_text)

    # Initialize classifier
    print(f"Initializing {model_type} classifier...")
    classifier = ClinicalNoteClassifier(model_type=model_type)

    # Train model
    print("Training model...")
    classifier.train(enhanced_texts, df["medical_specialty"].tolist())

    # Save model
    model_path = f"./results/classical_ml_enhanced/{model_type}_model.joblib"
    print(f"Saving model to {model_path}...")
    classifier.save_model(model_path)

    print("Training completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train enhanced classical ML model for clinical note classification"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to training data CSV file"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest"],
        help="Type of model to train (logistic or random_forest)",
    )

    args = parser.parse_args()

    train_model(args.data_path, args.model_type)
