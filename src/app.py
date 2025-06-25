import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os

# Set page config
st.set_page_config(
    page_title="Clinical Note Classifier",
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
st.title("🏥 Clinical Note Classifier")
st.markdown("""
This application uses a fine-tuned DistilBERT model to classify clinical notes into different categories.
Enter your clinical note text below and click 'Classify' to get the prediction.
""")

def predict(text, model, tokenizer):
    try:
        # Clean and prepare the input text
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty")
            
        # Tokenize the input text
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device  #Accesses the device attribute of that parameter, which will be either
        inputs = {k: v.to(device) for k, v in inputs.items()}  #iterates through the inputs dictionary 
                                                                #(which typically contains input_ids, attention_mask, etc
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        predictions = []
        for prob, idx in zip(top3_prob[0], top3_indices[0]):
            label = model.config.id2label[str(idx.item())]  # Convert idx to string for dictionary lookup
            predictions.append({
                "label": label,
                "probability": prob.item()
            })
        
        return predictions
    except Exception as e:
        st.error(f"Detailed error in prediction: {str(e)}")
        st.error(f"Model config: {model.config}")
        st.error(f"Input text length: {len(text)}")
        st.error(f"Input text: {text[:100]}...")  # Show first 100 chars of input
        raise e

@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Load the model and tokenizer
        model_path = "./results/best_model"
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Load label mappings
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)
            id2label = config["id2label"]
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(id2label),
            id2label=id2label
        )
        
        # Set model to evaluation mode
        model.eval()
        
        return model, tokenizer, id2label
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e

# Load model and tokenizer
try:
    model, tokenizer, id2label = load_model_and_tokenizer()
    st.success("Model loaded successfully!")
    # Display model info
    st.info(f"Model loaded with {len(id2label)} labels")
    st.json(id2label)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

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
    else:
        with st.spinner("Analyzing..."):
            try:
                predictions = predict(text_input, model, tokenizer)
                
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
    <p>Built with Streamlit and Hugging Face Transformers</p>
</div>
""", unsafe_allow_html=True) 