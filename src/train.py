import numpy as np
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from data_loader import load_and_prepare_data
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import time
import os
import shutil
import json

# --- Configuration ---
MODEL_CHECKPOINT = "distilbert-base-uncased"
DATA_PATH = "./data/mtsamples.csv"
OUTPUT_DIR = "./results"

# --- Load and Prepare Data ---
print("Loading and preparing data...")
raw_datasets, id2label = load_and_prepare_data(DATA_PATH, augment=True)
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

print(f"Data loaded. Train size: {len(raw_datasets['train'])}, Test size: {len(raw_datasets['test'])}")

# --- Tokenization ---
print("Tokenizing data...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# --- Model ---
print("Loading pre-trained model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# --- Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate precision, recall, and F1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Calculate macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    # Calculate accuracy
    accuracy = (predictions == labels).mean()
    
    # Create detailed metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
    }
    
    # Add per-class metrics
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        metrics[f'precision_class_{i}'] = p
        metrics[f'recall_class_{i}'] = r
        metrics[f'f1_class_{i}'] = f
    
    return metrics

# --- Training ---
print("Setting up training...")

class SafeCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        try:
            # Add a small delay before saving
            time.sleep(1)
            return control
        except Exception as e:
            print(f"Warning: Error during checkpoint save: {str(e)}")
            return control

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,  # Reduced frequency of saves
    load_best_model_at_end=True,
    metric_for_best_model="weighted_f1",
    greater_is_better=True,
    learning_rate=2e-5,
    fp16=True,
    gradient_accumulation_steps=2,
    save_safetensors=False,  # Disable safetensors format
    save_total_limit=2,  # Keep only the last 2 checkpoints
)

# Add early stopping and safe checkpoint callbacks
callbacks = [
    EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    ),
    SafeCheckpointCallback()
]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=callbacks
)

print("Starting training...")
trainer.train()
print("Training complete.")

# Save the model with error handling
def save_model_safely(trainer, output_dir):
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Create a temporary directory for saving
            temp_dir = f"{output_dir}_temp"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            # Save to temporary directory first
            trainer.save_model(temp_dir)
            
            # If successful, move to final location
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            shutil.move(temp_dir, output_dir)
            print(f"Model successfully saved to {output_dir}")
            return True
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to save model after all attempts")
                return False

# Save the best model
best_model_path = f"{OUTPUT_DIR}/best_model"
if save_model_safely(trainer, best_model_path):
    print(f"Best model saved to: {best_model_path}")
else:
    print("Warning: Could not save the best model. You may need to manually save it later.")

# Save training history
try:
    history = trainer.state.log_history
    with open(f"{OUTPUT_DIR}/training_history.json", 'w') as f:
        json.dump(history, f)
    print("Training history saved.")
except Exception as e:
    print(f"Warning: Could not save training history: {str(e)}")