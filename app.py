
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig # PeftModel is crucial for loading adapter
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# The LoRA adapter weights folder is small and is uploaded to GitHub
LORA_ADAPTER_PATH = "ft_lora_adapter" 
# The base model is downloaded from Hugging Face at runtime (big download, but only done once per deployment)
BASE_MODEL_NAME = "distilbert-base-uncased" 
LE_CLASSES_FILE = "le_classes.txt"

# Use st.cache_resource for heavy, one-time loading operations
@st.cache_resource
def load_assets():
    st.info(f"Loading base model: {BASE_MODEL_NAME} and adapter from {LORA_ADAPTER_PATH}...")
    
    try:
        # 1. Load the Label Encoder Classes
        with open(LE_CLASSES_FILE, 'r') as f:
            le_classes = f.read().split(',')
        le = LabelEncoder()
        le.fit(le_classes)
        label_classes = list(le.classes_)

        # 2. Load the base DistilBERT model from Hugging Face 
        # The number of labels MUST match what you trained with.
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, 
            num_labels=len(label_classes)
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        # 3. Load the LoRA adapter weights from the small GitHub folder and attach them to the base model
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        
        # Ensure model is on CPU for Streamlit Cloud
        device = torch.device("cpu")
        model.to(device)
        
    except Exception as e:
        st.error(f"Error loading model assets. Ensure the '{LORA_ADAPTER_PATH}' folder and '{LE_CLASSES_FILE}' are in your GitHub root. Details: {e}")
        return None, None, None, None
    
    return model, tokenizer, le, label_classes

model, tokenizer, label_encoder, label_classes = load_assets()

# -----------------
# START OF STREAMLIT APP UI
# -----------------
st.set_page_config(page_title="LoRA Bias Detector", layout="wide")
st.title("📰 LoRA News Headline Bias Detector")
st.markdown("Model uses DistilBERT base downloaded at runtime, fine-tuned with a **tiny LoRA adapter** from GitHub.")
st.markdown("---")

headline_input = st.text_area(
    "Enter a News Headline to Analyze:", 
    "The government's latest spending bill is a triumph for working families.", 
    height=150
)

if model and tokenizer: # Check if assets loaded successfully
    if st.button("Analyze Bias", use_container_width=True, type="primary"):
        if headline_input:
            with st.spinner('Analyzing bias...'):
                device = torch.device("cpu") 
                
                inputs = tokenizer(headline_input, return_tensors="pt", truncation=True, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    model.eval()
                    outputs = model(**inputs)
                
                logits = outputs.logits
                probabilities = logits.softmax(dim=1).cpu().numpy()[0]
                predicted_class_id = np.argmax(probabilities)
                predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
                
                st.success("Analysis Complete!")
                st.subheader(f"Predicted Bias: **{predicted_label}**")
                
                prob_df = pd.DataFrame({
                    'Bias': label_classes,
                    'Probability': probabilities
                })
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                prob_df['Percentage'] = (prob_df['Probability'] * 100).round(1).astype(str) + '%'

                st.markdown("#### Probability Distribution")
                st.bar_chart(prob_df.set_index('Bias')['Probability'])
                st.table(prob_df[['Bias', 'Percentage']].rename(columns={'Percentage': 'Confidence'}))

        else:
            st.warning("Please enter a headline to analyze.")
else:
    st.error("Model assets failed to load. Check your Colab steps and GitHub files.")
