from classical_ml_classifier import ClinicalNoteClassifier
import pandas as pd

def test_model(model_path):
    # Load the trained model
    classifier = ClinicalNoteClassifier()
    classifier.load_model(model_path)
    
    # Example clinical notes from different specialties
    test_notes = [
        {
            "note": "Patient presents with severe headache and photophobia. History of migraines. No recent trauma. Neurological exam normal except for mild photophobia.",
            "expected_specialty": "Neurology"
        },
        {
            "note": "Follow-up visit for type 2 diabetes. Blood glucose levels stable. HbA1c 6.8%. Continue current medication regimen.",
            "expected_specialty": "Endocrinology"
        },
        {
            "note": "Routine physical examination. All vital signs normal. No significant findings. Patient reports good overall health.",
            "expected_specialty": "General Medicine"
        },
        {
            "note": "Patient reports persistent cough and shortness of breath. Chest X-ray shows mild infiltrates. Prescribed antibiotics.",
            "expected_specialty": "Cardiovascular / Pulmonary"
        },
        {
            "note": "Pre-operative evaluation for total knee replacement. Patient cleared for surgery. No contraindications found.",
            "expected_specialty": "Orthopedic"
        }
    ]
    
    print("\nTesting model with example clinical notes:")
    print("=" * 80)
    
    for i, test_case in enumerate(test_notes, 1):
        print(f"\nTest Case {i}:")
        print(f"Note: {test_case['note']}")
        print(f"Expected Specialty: {test_case['expected_specialty']}")
        
        # Get predictions
        predictions = classifier.predict(test_case['note'])
        
        print("\nPredictions:")
        for j, pred in enumerate(predictions, 1):
            print(f"{j}. {pred['label']}: {pred['probability']:.2%}")
        
        print("-" * 80)

if __name__ == "__main__":
    # Path to the trained model
    model_path = "./results/classical_ml/logistic_model.joblib"
    
    try:
        test_model(model_path)
    except Exception as e:
        print(f"Error testing model: {str(e)}") 