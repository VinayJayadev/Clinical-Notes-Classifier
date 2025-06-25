import pandas as pd
from classical_ml_classifier import ClinicalNoteClassifier
from enhanced_data_loader import load_and_prepare_data_enhanced
import joblib
import os

def train_model_enhanced(data_path, model_type='logistic', use_keywords=True):
    """
    Train the classical ML model with enhanced data loading
    Args:
        data_path (str): Path to the training data CSV file
        model_type (str): Either 'logistic' or 'random_forest'
        use_keywords (bool): Whether to use keywords for enhancement
    """
    # Create results directory if it doesn't exist
    os.makedirs('./results/classical_ml', exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    print(f"Using keywords: {use_keywords}")
    
    # Use enhanced data loader
    dataset_dict, id2label = load_and_prepare_data_enhanced(
        data_path, 
        test_size=0.2, 
        augment=True, 
        use_keywords=use_keywords
    )
    
    # Convert to lists for classical ML
    train_texts = dataset_dict['train']['text']
    train_labels = dataset_dict['train']['label_str']
    
    # Initialize classifier
    print(f"Initializing {model_type} classifier...")
    classifier = ClinicalNoteClassifier(model_type=model_type)
    
    # Train model
    print("Training model...")
    classifier.train(train_texts, train_labels)
    
    # Save model
    model_suffix = "_with_keywords" if use_keywords else "_no_keywords"
    model_path = f"./results/classical_ml/{model_type}_model{model_suffix}.joblib"
    print(f"Saving model to {model_path}...")
    classifier.save_model(model_path)
    
    print("Training completed successfully!")
    return classifier, dataset_dict

def compare_keywords_performance(data_path, model_type='logistic'):
    """
    Compare performance with and without keywords
    """
    print("="*60)
    print("COMPARING KEYWORDS PERFORMANCE")
    print("="*60)
    
    # Train model without keywords
    print("\n🔍 Training WITHOUT keywords...")
    classifier_no_kw, dataset_no_kw = train_model_enhanced(
        data_path, model_type, use_keywords=False
    )
    
    # Train model with keywords
    print("\n🔍 Training WITH keywords...")
    classifier_with_kw, dataset_with_kw = train_model_enhanced(
        data_path, model_type, use_keywords=True
    )
    
    # Test both models
    test_texts = dataset_no_kw['test']['text']
    test_labels = dataset_no_kw['test']['label_str']
    
    print("\n📊 Performance Comparison:")
    print("-" * 40)
    
    # Test without keywords model
    correct_no_kw = 0
    for text, true_label in zip(test_texts, test_labels):
        pred = classifier_no_kw.predict(text)[0]['label']
        if pred == true_label:
            correct_no_kw += 1
    
    accuracy_no_kw = correct_no_kw / len(test_texts)
    print(f"Without keywords: {accuracy_no_kw:.3f} ({correct_no_kw}/{len(test_texts)})")
    
    # Test with keywords model
    correct_with_kw = 0
    for text, true_label in zip(test_texts, test_labels):
        pred = classifier_with_kw.predict(text)[0]['label']
        if pred == true_label:
            correct_with_kw += 1
    
    accuracy_with_kw = correct_with_kw / len(test_texts)
    print(f"With keywords:    {accuracy_with_kw:.3f} ({correct_with_kw}/{len(test_texts)})")
    
    # Calculate improvement
    improvement = accuracy_with_kw - accuracy_no_kw
    improvement_pct = (improvement / accuracy_no_kw) * 100 if accuracy_no_kw > 0 else 0
    
    print(f"\n📈 Improvement: {improvement:.3f} ({improvement_pct:+.1f}%)")
    
    if improvement > 0:
        print("✅ Keywords improved performance!")
    else:
        print("❌ Keywords did not improve performance")
    
    return {
        'without_keywords': accuracy_no_kw,
        'with_keywords': accuracy_with_kw,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train enhanced classical ML model for clinical note classification')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data CSV file')
    parser.add_argument('--model_type', type=str, default='logistic', choices=['logistic', 'random_forest'],
                      help='Type of model to train (logistic or random_forest)')
    parser.add_argument('--use_keywords', action='store_true', help='Use keywords for enhancement')
    parser.add_argument('--compare', action='store_true', help='Compare performance with and without keywords')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_keywords_performance(args.data_path, args.model_type)
    else:
        train_model_enhanced(args.data_path, args.model_type, args.use_keywords) 