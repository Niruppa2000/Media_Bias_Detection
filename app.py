## --- 8. STREAMLIT DEPLOYMENT SETUP AND AUTO-DOWNLOAD ---
import os
import subprocess
import nest_asyncio
from google.colab import files

# --- 1. Save Model and Tokenizer ---
print("Saving model and tokenizer...")
ft_model.save_pretrained("./ft_bias_model")
ft_tokenizer.save_pretrained("./ft_bias_model")

# --- 2. Save Label Encoder Classes ---
# This file is critical for mapping the model's output IDs back to "Left", "Right", "Neutral".
le_classes = list(label_encoder.classes_)
with open('le_classes.txt', 'w') as f:
    f.write(','.join(le_classes))
print(f"Saved label classes: {le_classes}")

# --- 3. Create app.py and requirements.txt (as provided in previous step) ---
streamlit_code = f"""
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

MODEL_PATH = "ft_bias_model"
LE_CLASSES_FILE = "le_classes.txt"

# --- Load Model and Assets ---
@st.cache_resource
def load_assets():
    try:
        with open(LE_CLASSES_FILE, 'r') as f:
            le_classes = f.read().split(',')
    except FileNotFoundError:
        st.error(f"Error: {{LE_CLASSES_FILE}} not found. Ensure it's uploaded to your GitHub repository.")
        return None, None, None, None

    le = LabelEncoder()
    le.fit(le_classes)
    label_classes = list(le.classes_)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # Force model to CPU if no GPU available in deployment environment
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model/tokenizer. Details: {{e}}")
        return None, None, None, None
    
    return model, tokenizer, le, label_classes

model, tokenizer, label_encoder, label_classes = load_assets()

if model is None:
    st.stop()
    
# Move model to CPU after loading if it's on a device not compatible with Streamlit hosting
device = torch.device("cpu")
model.to(device)

# --- Prediction Function ---
def predict_bias(headline, model, tokenizer, label_encoder):
    
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True)
    inputs = {{k: v.to(device) for k, v in inputs.items()}}

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = logits.softmax(dim=1).cpu().numpy()[0]
    predicted_class_id = np.argmax(probabilities)
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    
    return predicted_label, probabilities

# --- Streamlit App UI ---
st.set_page_config(page_title="News Bias Detector", layout="wide")
st.title("📰 News Headline Bias Detector")
st.markdown("Use a fine-tuned DistilBERT model to classify headline bias as **Left**, **Right**, or **Neutral**.")
st.markdown("---")

headline_input = st.text_area(
    "Enter a News Headline to Analyze:", 
    "The government's latest spending bill is a triumph for working families.", 
    height=150
)

if st.button("Analyze Bias", use_container_width=True, type="primary"):
    if headline_input:
        with st.spinner('Analyzing bias...'):
            predicted_bias, probabilities = predict_bias(headline_input, model, tokenizer, label_encoder)
            
            st.success("Analysis Complete!")
            
            st.subheader(f"Predicted Bias: **{{predicted_bias}}**")
            
            prob_df = pd.DataFrame({{
                'Bias': label_classes,
                'Probability': probabilities
            }})
            prob_df = prob_df.sort_values(by='Probability', ascending=False)
            prob_df['Percentage'] = (prob_df['Probability'] * 100).round(1).astype(str) + '%'

            st.markdown("#### Probability Distribution")
            st.bar_chart(prob_df.set_index('Bias')['Probability'])
            st.table(prob_df[['Bias', 'Percentage']].rename(columns={{'Percentage': 'Confidence'}}))

    else:
        st.warning("Please enter a headline to analyze.")
"""

with open("app.py", "w") as f:
    f.write(streamlit_code)

# Create requirements.txt
requirements_content = """streamlit
pandas
numpy
scikit-learn
torch
transformers
datasets
sentence-transformers
"""
with open("requirements.txt", "w") as f:
    f.write(requirements_content)
print("Created app.py and requirements.txt.")

# --- 4. Auto-Download the Zipped Model Artifacts ---
print("\nZipping model artifacts...")
!zip -r ft_bias_model_deployment.zip ft_bias_model/ le_classes.txt app.py requirements.txt

print("Triggering automatic download of 'ft_bias_model_deployment.zip'...")
try:
    files.download('ft_bias_model_deployment.zip')
    print("Download initiated. Check your browser's downloads.")
except Exception as e:
    print("Download failed. Please manually download the 'ft_bias_model_deployment.zip' file from the Colab file browser (left sidebar).")
