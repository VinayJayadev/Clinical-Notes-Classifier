from data_loader import load_and_prepare_data
from datasets import DatasetDict
import os

def test_load_and_prepare_data():
    """
    Tests the data loading and preparation function.
    For a true unit test, you would create a small dummy csv in the tests folder.
    For simplicity, we check if the function runs on the actual data file.
    """
    filepath = "./data/mtsamples.csv"
    
    # Ensure the data file exists before running the test
    if not os.path.exists(filepath):
        # Create a dummy file for the test to pass in a CI/CD environment
        os.makedirs("./data", exist_ok=True)
        with open(filepath, "w") as f:
            f.write("id,transcription,medical_specialty\n")
            f.write("1,some text,Cardiovascular / Pulmonary\n")
            f.write("2,more text,Neurology\n")

    dataset_dict, id2label = load_and_prepare_data(filepath)

    assert isinstance(dataset_dict, DatasetDict)
    assert "train" in dataset_dict
    assert "test" in dataset_dict
    assert "text" in dataset_dict["train"].features
    assert "label" in dataset_dict["train"].features
    assert isinstance(id2label, dict)