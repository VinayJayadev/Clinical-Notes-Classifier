import streamlit as st
import pandas as pd
from classical_ml_classifier import ClinicalNoteClassifier
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Clinical Note Classifier (Classical ML)",
    page_icon="🏥",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        height: 200px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏥 Clinical Note Classifier (Classical ML)")
st.markdown("""
This application uses classical machine learning models to classify clinical notes into different categories.
Enter your clinical note text below and click 'Classify' to get the prediction.
""")

@st.cache_resource
def load_model(model_type):
    try:
        model_path = f"./results/classical_ml/{model_type}_model.joblib"
        if os.path.exists(model_path):
            classifier = ClinicalNoteClassifier(model_type=model_type)
            classifier.load_model(model_path)
            return classifier
        else:
            st.warning(f"No trained {model_type} model found. Please train the model first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Model type selection
model_type = st.selectbox(
    "Select Model Type",
    ["logistic", "random_forest"],
    format_func=lambda x: "Logistic Regression" if x == "logistic" else "Random Forest",
    index=0
)

# Load selected model
classifier = load_model(model_type)

# Text input
text_input = st.text_area(
    "Enter your clinical note:",
    placeholder="Type or paste your clinical note here...",
    height=200
)

# Classify button
if st.button("Classify", type="primary"):
    if text_input.strip() == "":
        st.warning("Please enter some text to classify.")
    elif classifier is None:
        st.error("No trained model available. Please train the model first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                predictions = classifier.predict(text_input)
                
                # Display predictions
                st.markdown("### Predictions")
                for i, pred in enumerate(predictions, 1):
                    prob_percentage = pred["probability"] * 100
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h4>{i}. {pred['label']}</h4>
                        <p>Confidence: {prob_percentage:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.error("Please try again with different text or contact support if the issue persists.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit and scikit-learn</p>
</div>
""", unsafe_allow_html=True) 