import json
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from classical_ml_classifier import ClinicalNoteClassifier


def load_and_prepare_data(data_path):
    """Load and prepare the data for evaluation"""
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Remove rows with missing values
    df = df.dropna(subset=["transcription", "medical_specialty"])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["transcription"],
        df["medical_specialty"],
        test_size=0.2,
        random_state=42,
        stratify=df["medical_specialty"],
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and return metrics"""
    print(f"\nEvaluating {model_name}...")

    # Get predictions
    y_pred = []
    y_pred_proba = []

    for text in X_test:
        predictions = model.predict(text)
        y_pred.append(predictions[0]["label"])
        y_pred_proba.append(predictions[0]["probability"])

    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate per-class metrics
    class_metrics = {}
    for label in sorted(set(y_test)):
        class_metrics[label] = {
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1-score": report[label]["f1-score"],
            "support": report[label]["support"],
        }

    return {
        "model_name": model_name,
        "accuracy": report["accuracy"],
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "weighted_avg_precision": report["weighted avg"]["precision"],
        "weighted_avg_recall": report["weighted avg"]["recall"],
        "weighted_avg_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm,
        "predictions": y_pred,
        "probabilities": y_pred_proba,
        "true_labels": y_test,
        "class_metrics": class_metrics,
    }


def plot_confusion_matrix(cm, model_name, labels):
    """Plot confusion matrix"""
    plt.figure(figsize=(15, 15))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45)
    plt.tight_layout()

    # Save to file
    plt.savefig(f"./results/classical_ml/{model_name}_confusion_matrix.png")

    # Log to MLflow
    mlflow.log_figure(
        plt.gcf(), f"confusion_matrices/{model_name}_confusion_matrix.png"
    )

    plt.close()


def plot_metrics_comparison(metrics):
    """Plot comparison of metrics between models"""
    metrics_df = pd.DataFrame(metrics)

    # Plot overall metrics
    plt.figure(figsize=(15, 6))
    x = np.arange(len(metrics_df))
    width = 0.2

    plt.bar(x - width * 1.5, metrics_df["accuracy"], width, label="Accuracy")
    plt.bar(
        x - width * 0.5,
        metrics_df["macro_avg_precision"],
        width,
        label="Macro Avg Precision",
    )
    plt.bar(
        x + width * 0.5, metrics_df["macro_avg_recall"], width, label="Macro Avg Recall"
    )
    plt.bar(x + width * 1.5, metrics_df["macro_avg_f1"], width, label="Macro Avg F1")

    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.xticks(x, metrics_df["model_name"])
    plt.legend()
    plt.tight_layout()

    # Save to file
    plt.savefig("./results/classical_ml/model_comparison.png")

    # Log to MLflow
    mlflow.log_figure(plt.gcf(), "visualizations/model_comparison.png")

    plt.close()


def plot_class_performance(metrics):
    """Plot performance metrics for each class"""
    for metric in metrics:
        model_name = metric["model_name"]
        class_metrics = pd.DataFrame(metric["class_metrics"]).T

        plt.figure(figsize=(15, 6))
        class_metrics[["precision", "recall", "f1-score"]].plot(kind="bar")
        plt.title(f"Class-wise Performance - {model_name}")
        plt.xlabel("Medical Specialty")
        plt.ylabel("Score")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        # Save to file
        plt.savefig(f"./results/classical_ml/{model_name}_class_performance.png")

        # Log to MLflow
        mlflow.log_figure(
            plt.gcf(), f"visualizations/{model_name}_class_performance.png"
        )

        plt.close()


def plot_prediction_differences(metrics):
    """Plot differences in predictions between models"""
    # Create comparison DataFrame
    comparison = pd.DataFrame(
        {
            "True Label": metrics[0]["true_labels"],
            "Logistic Regression": metrics[0]["predictions"],
            "Random Forest": metrics[1]["predictions"],
        }
    )

    # Calculate agreement
    agreement = (
        comparison["Logistic Regression"] == comparison["Random Forest"]
    ).mean()

    # Plot agreement by class
    plt.figure(figsize=(15, 6))
    agreement_by_class = (
        comparison.groupby("True Label")
        .apply(lambda x: (x["Logistic Regression"] == x["Random Forest"]).mean())
        .sort_values(ascending=False)
    )

    agreement_by_class.plot(kind="bar")
    plt.title(f"Model Agreement by Class (Overall Agreement: {agreement:.2%})")
    plt.xlabel("Medical Specialty")
    plt.ylabel("Agreement Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save to file
    plt.savefig("./results/classical_ml/model_agreement.png")

    # Log to MLflow
    mlflow.log_figure(plt.gcf(), "visualizations/model_agreement.png")

    plt.close()


def generate_report(metrics):
    """Generate a detailed report of model performance"""
    report = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "models": {}}

    for metric in metrics:
        model_name = metric["model_name"]
        report["models"][model_name] = {
            "overall_metrics": {
                "accuracy": float(metric["accuracy"]),
                "macro_avg_precision": float(metric["macro_avg_precision"]),
                "macro_avg_recall": float(metric["macro_avg_recall"]),
                "macro_avg_f1": float(metric["macro_avg_f1"]),
                "weighted_avg_precision": float(metric["weighted_avg_precision"]),
                "weighted_avg_recall": float(metric["weighted_avg_recall"]),
                "weighted_avg_f1": float(metric["weighted_avg_f1"]),
            },
            "class_metrics": metric["class_metrics"],
        }

    # Save report
    with open("./results/classical_ml/model_comparison_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Generate markdown report
    md_report = f"""# Model Comparison Report
Generated on: {report['timestamp']}

## Overall Performance

"""

    for model_name, model_metrics in report["models"].items():
        md_report += f"""
### {model_name}

#### Overall Metrics
- Accuracy: {model_metrics['overall_metrics']['accuracy']:.3f}
- Macro Average Precision: {model_metrics['overall_metrics']['macro_avg_precision']:.3f}
- Macro Average Recall: {model_metrics['overall_metrics']['macro_avg_recall']:.3f}
- Macro Average F1-Score: {model_metrics['overall_metrics']['macro_avg_f1']:.3f}
- Weighted Average Precision: {model_metrics['overall_metrics']['weighted_avg_precision']:.3f}
- Weighted Average Recall: {model_metrics['overall_metrics']['weighted_avg_recall']:.3f}
- Weighted Average F1-Score: {model_metrics['overall_metrics']['weighted_avg_f1']:.3f}

#### Class-wise Performance
| Medical Specialty | Precision | Recall | F1-Score | Support |
|------------------|-----------|---------|-----------|----------|
"""

        for specialty, metrics in model_metrics["class_metrics"].items():
            md_report += f"| {specialty} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1-score']:.3f} | {metrics['support']} |\n"

    # Save markdown report
    with open("./results/classical_ml/model_comparison_report.md", "w") as f:
        f.write(md_report)


def setup_mlflow():
    """Setup MLflow tracking"""
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")

    # Create or get experiment
    experiment_name = "Clinical Note Classification"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_metrics_to_mlflow(metrics):
    """Log metrics to MLflow"""
    for metric in metrics:
        model_name = metric["model_name"]

        # Log overall metrics
        mlflow.log_metric(f"{model_name}_accuracy", metric["accuracy"])
        mlflow.log_metric(
            f"{model_name}_macro_avg_precision", metric["macro_avg_precision"]
        )
        mlflow.log_metric(f"{model_name}_macro_avg_recall", metric["macro_avg_recall"])
        mlflow.log_metric(f"{model_name}_macro_avg_f1", metric["macro_avg_f1"])
        mlflow.log_metric(
            f"{model_name}_weighted_avg_precision", metric["weighted_avg_precision"]
        )
        mlflow.log_metric(
            f"{model_name}_weighted_avg_recall", metric["weighted_avg_recall"]
        )
        mlflow.log_metric(f"{model_name}_weighted_avg_f1", metric["weighted_avg_f1"])

        # Log class-wise metrics
        for label, class_metric in metric["class_metrics"].items():
            mlflow.log_metric(
                f"{model_name}_{label}_precision", class_metric["precision"]
            )
            mlflow.log_metric(f"{model_name}_{label}_recall", class_metric["recall"])
            mlflow.log_metric(f"{model_name}_{label}_f1", class_metric["f1-score"])


def log_artifacts_to_mlflow(metrics):
    """Log artifacts to MLflow"""
    # Log confusion matrices
    for metric in metrics:
        model_name = metric["model_name"]
        cm_path = f"./results/classical_ml/{model_name}_confusion_matrix.png"
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, f"confusion_matrices/{model_name}")

    # Log other visualizations
    if os.path.exists("./results/classical_ml/model_comparison.png"):
        mlflow.log_artifact(
            "./results/classical_ml/model_comparison.png", "visualizations"
        )
    if os.path.exists("./results/classical_ml/model_agreement.png"):
        mlflow.log_artifact(
            "./results/classical_ml/model_agreement.png", "visualizations"
        )

    # Log reports
    if os.path.exists("./results/classical_ml/model_comparison_report.json"):
        mlflow.log_artifact(
            "./results/classical_ml/model_comparison_report.json", "reports"
        )
    if os.path.exists("./results/classical_ml/model_comparison_report.md"):
        mlflow.log_artifact(
            "./results/classical_ml/model_comparison_report.md", "reports"
        )


def log_model_to_mlflow(model, model_name):
    """Log model to MLflow"""
    # Log the model
    mlflow.sklearn.log_model(
        model.pipeline,
        f"{model_name}_model",
        registered_model_name=f"clinical_note_{model_name}",
        signature=infer_signature(
            model.pipeline.named_steps["tfidf"].transform(["sample text"]),
            model.pipeline.predict(["sample text"]),
        ),
    )


def main():
    # Setup MLflow
    experiment_id = setup_mlflow()

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id

        # Set run tags
        mlflow.set_tag("project", "Clinical Note Classification")
        mlflow.set_tag("task", "Multi-class Classification")
        mlflow.set_tag("evaluation_type", "Model Comparison")
        mlflow.set_tag("framework", "scikit-learn")

        # Log parameters
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("dataset_path", "./mtsamples.csv")
        mlflow.log_param("models_evaluated", "Logistic Regression, Random Forest")
        mlflow.log_param(
            "evaluation_metrics", "accuracy,precision,recall,f1,confusion_matrix"
        )

        # Create results directory
        os.makedirs("./results/classical_ml", exist_ok=True)

        # Load data
        X_train, X_test, y_train, y_test = load_and_prepare_data("./mtsamples.csv")

        # Log dataset info
        mlflow.log_param("n_samples", len(X_train) + len(X_test))
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("n_classes", len(set(y_test)))
        mlflow.log_param("classes", list(sorted(set(y_test))))

        # Log dataset statistics
        class_distribution = y_train.value_counts().to_dict()
        mlflow.log_dict(class_distribution, "dataset/class_distribution.json")

        # Log dataset as artifact
        if os.path.exists("./mtsamples.csv"):
            mlflow.log_artifact("./mtsamples.csv", "dataset")

        # Load models
        models = {
            "Logistic Regression": "./results/classical_ml/logistic_model.joblib",
            "Random Forest": "./results/classical_ml/random_forest_model.joblib",
        }

        # Evaluate models
        metrics = []
        for model_name, model_path in models.items():
            if os.path.exists(model_path):
                # Log model file as artifact
                mlflow.log_artifact(
                    model_path, f"models/{model_name.lower().replace(' ', '_')}"
                )

                model = ClinicalNoteClassifier()
                model.load_model(model_path)

                # Log model hyperparameters
                if (
                    hasattr(model.pipeline, "named_steps")
                    and "classifier" in model.pipeline.named_steps
                ):
                    classifier = model.pipeline.named_steps["classifier"]
                    if hasattr(classifier, "get_params"):
                        params = classifier.get_params()
                        for param_name, param_value in params.items():
                            mlflow.log_param(
                                f"{model_name.lower().replace(' ', '_')}_{param_name}",
                                param_value,
                            )

                model_metrics = evaluate_model(model, X_test, y_test, model_name)
                metrics.append(model_metrics)

                # Plot confusion matrix
                plot_confusion_matrix(
                    model_metrics["confusion_matrix"], model_name, sorted(set(y_test))
                )

                # Log model to MLflow
                log_model_to_mlflow(model, model_name.lower().replace(" ", "_"))

        # Plot comparisons
        if len(metrics) > 1:
            plot_metrics_comparison(metrics)
            plot_class_performance(metrics)
            plot_prediction_differences(metrics)
            generate_report(metrics)

            # Log metrics and artifacts to MLflow
            log_metrics_to_mlflow(metrics)
            log_artifacts_to_mlflow(metrics)

            # Log complete evaluation results
            evaluation_summary = {
                "timestamp": datetime.now().isoformat(),
                "dataset_info": {
                    "n_samples": len(X_train) + len(X_test),
                    "n_train_samples": len(X_train),
                    "n_test_samples": len(X_test),
                    "n_classes": len(set(y_test)),
                    "classes": list(sorted(set(y_test))),
                },
                "models_evaluated": [metric["model_name"] for metric in metrics],
                "results": {},
            }

            for metric in metrics:
                evaluation_summary["results"][metric["model_name"]] = {
                    "overall_metrics": {
                        "accuracy": float(metric["accuracy"]),
                        "macro_avg_precision": float(metric["macro_avg_precision"]),
                        "macro_avg_recall": float(metric["macro_avg_recall"]),
                        "macro_avg_f1": float(metric["macro_avg_f1"]),
                        "weighted_avg_precision": float(
                            metric["weighted_avg_precision"]
                        ),
                        "weighted_avg_recall": float(metric["weighted_avg_recall"]),
                        "weighted_avg_f1": float(metric["weighted_avg_f1"]),
                    },
                    "class_metrics": metric["class_metrics"],
                }

            mlflow.log_dict(evaluation_summary, "evaluation_results.json")

            # Print detailed comparison
            print("\nModel Performance Comparison:")
            print("=" * 80)
            for metric in metrics:
                print(f"\n{metric['model_name']}:")
                print(f"Accuracy: {metric['accuracy']:.3f}")
                print(f"Macro Avg Precision: {metric['macro_avg_precision']:.3f}")
                print(f"Macro Avg Recall: {metric['macro_avg_recall']:.3f}")
                print(f"Macro Avg F1-Score: {metric['macro_avg_f1']:.3f}")
                print(f"Weighted Avg Precision: {metric['weighted_avg_precision']:.3f}")
                print(f"Weighted Avg Recall: {metric['weighted_avg_recall']:.3f}")
                print(f"Weighted Avg F1-Score: {metric['weighted_avg_f1']:.3f}")

        print(f"\nMLflow run ID: {run_id}")
        print("You can view the results in the MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    main()
