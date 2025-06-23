import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_loader import load_and_prepare_data
import numpy as np # Import numpy

# --- Configuration ---
MODEL_PATH = "./results/best_model" 
DATA_PATH = "./data/mtsamples.csv"

# --- Load Model and Tokenizer ---
print(f"Loading fine-tuned model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Load Test Data ---
raw_datasets, _ = load_and_prepare_data(DATA_PATH)
test_dataset = raw_datasets['test']
true_labels = test_dataset['label']

# --- Make Predictions ---
print("Running predictions on the test set...")
predictions = []
for text in test_dataset['text']:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predictions.append(predicted_class_id)

# --- Display Results ---

# --- FIX IS HERE ---
# Find the unique labels that are actually present in the true labels and predictions
active_labels = np.union1d(true_labels, predictions)
# Create a list of target names that corresponds *only* to the active labels
active_target_names = [model.config.id2label[label] for label in active_labels]
# --- END OF FIX ---

print("\n--- Classification Report ---")
# Update the function call to use the 'labels' and new 'target_names' parameters
print(classification_report(true_labels, predictions, labels=active_labels, target_names=active_target_names))


print("\n--- Confusion Matrix ---")
cm = confusion_matrix(true_labels, predictions, labels=active_labels)
cm_df = pd.DataFrame(cm, index=active_target_names, columns=active_target_names)
print(cm_df)