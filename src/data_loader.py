import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict
from sklearn.utils import resample


def augment_text(text):
    """
    Simple text augmentation techniques.
    """
    # Add common medical abbreviations
    replacements = {
        "chest pain": "CP",
        "shortness of breath": "SOB",
        "electrocardiogram": "EKG",
        "magnetic resonance imaging": "MRI",
        "computed tomography": "CT",
    }

    augmented = text
    for original, replacement in replacements.items():
        if original in text.lower():
            augmented = augmented.replace(original, replacement)

    return augmented


def balance_dataset(df, min_samples=10):
    """
    Balance the dataset by upsampling minority classes.
    """
    # Get the count of samples per class
    class_counts = df["label_str"].value_counts()

    # Find classes that need upsampling
    classes_to_upsample = class_counts[class_counts < min_samples].index

    # Upsample each minority class
    dfs_upsampled = []
    for cls in classes_to_upsample:
        df_class = df[df["label_str"] == cls]
        df_upsampled = resample(
            df_class, replace=True, n_samples=min_samples, random_state=42
        )
        dfs_upsampled.append(df_upsampled)

    # Combine original and upsampled data
    df_balanced = pd.concat([df] + dfs_upsampled)
    return df_balanced


def load_and_prepare_data(filepath: str, test_size: float = 0.2, augment: bool = True):
    """
    Loads the medical transcriptions dataset, cleans it,
    and splits it into training and testing sets.

    Args:
        filepath (str): The path to the mtsamples.csv file.
        test_size (float): The proportion of the dataset to allocate to the test split.
        augment (bool): Whether to apply data augmentation.

    Returns:
        A Hugging Face DatasetDict with 'train' and 'test' splits, and a label mapping dict.
    """
    df = pd.read_csv(filepath)

    # Print initial class distribution
    print("\nInitial class distribution:")
    print(df["medical_specialty"].value_counts())

    df.dropna(subset=["transcription", "medical_specialty"], inplace=True)
    df.drop_duplicates(subset=["transcription"], inplace=True)

    df = df.rename(columns={"transcription": "text", "medical_specialty": "label_str"})

    df = df[["text", "label_str"]]

    # Apply data augmentation if requested
    if augment:
        print("\nApplying data augmentation...")
        augmented_texts = df["text"].apply(augment_text)
        df_augmented = df.copy()
        df_augmented["text"] = augmented_texts
        df = pd.concat([df, df_augmented], ignore_index=True)

    # Balance the dataset
    print("\nBalancing dataset...")
    df = balance_dataset(df, min_samples=20)

    # This creates the integer labels and the list of class names
    df["label"], class_names = pd.factorize(df["label_str"])

    id2label = dict(enumerate(class_names))

    dataset = Dataset.from_pandas(df)

    # We now explicitly create a ClassLabel feature type from our list of class names.
    class_label_feature = ClassLabel(names=list(class_names))
    dataset = dataset.cast_column("label", class_label_feature)

    # Split the data
    train_test_split = dataset.train_test_split(
        test_size=test_size, stratify_by_column="label"
    )

    dataset_dict = DatasetDict(
        {"train": train_test_split["train"], "test": train_test_split["test"]}
    )

    return dataset_dict, id2label
