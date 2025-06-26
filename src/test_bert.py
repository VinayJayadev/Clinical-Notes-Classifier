import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_loader import load_and_prepare_data


def test_bert_model(
    model_path="./results/best_model", data_path="./data/mtsamples.csv"
):
    """
    Test the trained BERT model and generate comprehensive evaluation metrics
    """
    print("=" * 60)
    print("BERT MODEL EVALUATION")
    print("=" * 60)

    # Load data
    print("Loading test data...")
    raw_datasets, id2label = load_and_prepare_data(data_path, augment=True)
    test_dataset = raw_datasets["test"]

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(id2label)}")

    # Load model and tokenizer
    print("Loading BERT model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set model to evaluation mode
    model.eval()

    # Prepare test data
    print("Preparing test data for evaluation...")
    test_texts = test_dataset["text"]
    test_labels = test_dataset["label"]

    # Tokenize test data
    def tokenize_function(examples):
        return tokenizer(
            examples,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    # Make predictions
    print("Making predictions...")
    predictions = []
    true_labels = []

    batch_size = 16
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i : i + batch_size]
        batch_labels = test_labels[i : i + batch_size]

        # Tokenize batch
        inputs = tokenize_function(batch_texts)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = model.to(device)

        # Get predictions
        with torch.no_grad():  # disables gradient computation, reducing memory usage and speeding up inference ( no updation of weights)
            outputs = model(**inputs)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        predictions.extend(batch_predictions)
        true_labels.extend(batch_labels)

    # Convert predictions and labels to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Calculate metrics
    print("Calculating metrics...")
    accuracy = accuracy_score(true_labels, predictions)

    # Generate classification report
    class_names = list(id2label.values())
    report = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Calculate macro and weighted averages
    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_precision = report["weighted avg"]["precision"]
    weighted_recall = report["weighted avg"]["recall"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    # Print results
    print("\n" + "=" * 60)
    print("BERT MODEL PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro Average Precision: {macro_precision:.4f}")
    print(f"Macro Average Recall: {macro_recall:.4f}")
    print(f"Macro Average F1-Score: {macro_f1:.4f}")
    print(f"Weighted Average Precision: {weighted_precision:.4f}")
    print(f"Weighted Average Recall: {weighted_recall:.4f}")
    print(f"Weighted Average F1-Score: {weighted_f1:.4f}")

    # Create results directory
    os.makedirs("./results/bert_evaluation", exist_ok=True)

    # Save detailed results
    results = {
        "model": "DistilRoBERTa (BERT)",
        "evaluation_date": datetime.now().isoformat(),
        "test_samples": len(test_dataset),
        "num_classes": len(id2label),
        "metrics": {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
        },
        "classification_report": report,
    }

    # Save results to JSON
    with open("./results/bert_evaluation/bert_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate and save classification report
    report_text = classification_report(
        true_labels, predictions, target_names=class_names, zero_division=0
    )

    with open("./results/bert_evaluation/bert_classification_report.txt", "w") as f:
        f.write("BERT Model Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n")
        f.write(f"Number of Classes: {len(id2label)}\n\n")
        f.write(report_text)

    # Create confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("BERT Model Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        "./results/bert_evaluation/bert_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create class performance visualization
    print("Generating class performance visualization...")
    class_metrics = []

    # Extract class metrics from the report
    for class_name, metrics in report.items():
        if class_name not in ["macro avg", "weighted avg", "accuracy"]:
            class_metrics.append(
                {
                    "class": class_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1-score"],
                    "support": metrics["support"],
                }
            )

    # Sort by F1 score
    class_metrics.sort(key=lambda x: x["f1"], reverse=True)

    # Plot top 20 performing classes
    top_classes = class_metrics[:20]

    if len(top_classes) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # F1 scores
        classes = [m["class"] for m in top_classes]
        f1_scores = [m["f1"] for m in top_classes]

        bars1 = ax1.barh(classes, f1_scores, color="skyblue")
        ax1.set_xlabel("F1 Score")
        ax1.set_title("BERT Model - Top 20 Classes by F1 Score")
        ax1.set_xlim(0, 1)

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars1, f1_scores)):
            ax1.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
                fontsize=9,
            )

        # Precision vs Recall
        precisions = [m["precision"] for m in top_classes]
        recalls = [m["recall"] for m in top_classes]

        ax2.scatter(precisions, recalls, s=100, alpha=0.7, c="red")

        # Only label top 10 classes to avoid overlapping
        for i, class_name in enumerate(classes[:10]):
            ax2.annotate(
                class_name,
                (precisions[i], recalls[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        ax2.set_xlabel("Precision")
        ax2.set_ylabel("Recall")
        ax2.set_title("BERT Model - Precision vs Recall (Top 20 Classes)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        # Add legend for unlabeled points
        if len(classes) > 10:
            ax2.text(
                0.02,
                0.98,
                f"Top 10 classes labeled\n{len(classes)-10} additional classes shown as points",
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(
            "./results/bert_evaluation/bert_class_performance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        print("Warning: No class metrics available for visualization")

    # Create summary markdown report
    print("Generating summary report...")
    markdown_report = f"""# BERT Model Evaluation Report

**Model**: DistilRoBERTa (BERT)
**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Samples**: {len(test_dataset)}
**Number of Classes**: {len(id2label)}

## Overall Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | {accuracy:.4f} ({accuracy*100:.2f}%) |
| **Macro Average Precision** | {macro_precision:.4f} |
| **Macro Average Recall** | {macro_recall:.4f} |
| **Macro Average F1-Score** | {macro_f1:.4f} |
| **Weighted Average Precision** | {weighted_precision:.4f} |
| **Weighted Average Recall** | {weighted_recall:.4f} |
| **Weighted Average F1-Score** | {weighted_f1:.4f} |

## Top 10 Performing Classes

| Rank | Class | F1 Score | Precision | Recall | Support |
|------|-------|----------|-----------|---------|---------|
"""

    for i, metric in enumerate(class_metrics[:10]):
        markdown_report += f"| {i+1} | {metric['class']} | {metric['f1']:.3f} | {metric['precision']:.3f} | {metric['recall']:.3f} | {metric['support']} |\n"

    markdown_report += f"""
## Bottom 10 Performing Classes

| Rank | Class | F1 Score | Precision | Recall | Support |
|------|-------|----------|-----------|---------|---------|
"""

    for i, metric in enumerate(class_metrics[-10:]):
        rank = len(class_metrics) - 9 + i
        markdown_report += f"| {rank} | {metric['class']} | {metric['f1']:.3f} | {metric['precision']:.3f} | {metric['recall']:.3f} | {metric['support']} |\n"

    markdown_report += """
## Key Insights

1. **Overall Performance**: The BERT model achieves high accuracy on the test set
2. **Class Imbalance**: Performance varies significantly across medical specialties
3. **Top Performers**: Classes with distinct medical terminology perform well
4. **Challenging Classes**: Rare specialties and those with overlapping terminology struggle

## Files Generated

- `bert_test_results.json`: Detailed metrics in JSON format
- `bert_classification_report.txt`: Full classification report
- `bert_confusion_matrix.png`: Confusion matrix visualization
- `bert_class_performance.png`: Class-wise performance plots
"""

    with open("./results/bert_evaluation/bert_evaluation_report.md", "w") as f:
        f.write(markdown_report)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("Results saved to: ./results/bert_evaluation/")
    print("- bert_test_results.json")
    print("- bert_classification_report.txt")
    print("- bert_confusion_matrix.png")
    print("- bert_class_performance.png")
    print("- bert_evaluation_report.md")

    return results


if __name__ == "__main__":
    test_bert_model()
