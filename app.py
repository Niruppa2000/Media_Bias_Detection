import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Define MODEL_PATH relative to the app.py file
MODEL_PATH = "ft_bias_model"
LE_CLASSES_FILE = "le_classes.txt"

# --- Load Model and Assets ---
# Use st.cache_resource to load heavy models once
@st.cache_resource
def load_assets():
    # 1. Load Label Encoder Classes
    try:
        with open(LE_CLASSES_FILE, 'r') as f:
            le_classes = f.read().split(',')
    except FileNotFoundError:
        st.error(f"Error: {LE_CLASSES_FILE} not found. Ensure it's uploaded to your GitHub repository.")
        return None, None, None, None

    le = LabelEncoder()
    le.fit(le_classes)
    label_classes = list(le.classes_)

    # 2. Load Model and Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {MODEL_PATH}. Ensure the directory and its files are uploaded. Details: {e}")
        return None, None, None, None
    
    return model, tokenizer, le, label_classes

model, tokenizer, label_encoder, label_classes = load_assets()

if model is None:
    st.stop()

# --- Prediction Function ---
def predict_bias(headline, model, tokenizer, label_encoder):
    # Ensure model is on CPU if not using a GPU environment
    device = torch.device("cpu") 
    model.to(device)
    
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

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
            
            # Display Prediction
            st.subheader(f"Predicted Bias: **{predicted_bias}**")
            
            # Create a DataFrame for the probabilities
            prob_df = pd.DataFrame({
                'Bias': label_classes,
                'Probability': probabilities
            })
            prob_df = prob_df.sort_values(by='Probability', ascending=False)
            prob_df['Percentage'] = (prob_df['Probability'] * 100).round(1).astype(str) + '%'

            # Display Probabilities
            st.markdown("#### Probability Distribution")
            
            # Display chart
            st.bar_chart(prob_df.set_index('Bias')['Probability'])
            
            # Display table
            st.table(prob_df[['Bias', 'Percentage']].rename(columns={'Percentage': 'Confidence'}))

    else:
        st.warning("Please enter a headline to analyze.")
