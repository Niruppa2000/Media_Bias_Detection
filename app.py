import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import numpy as np
import pandas as pd
import os

LORA_ADAPTER_PATH = "ft_lora_adapter"
BASE_MODEL_NAME = "distilbert-base-uncased"
LE_CLASSES_FILE = "le_classes.txt"

@st.cache_resource
def load_assets():
    st.info(f"Loading base model: {BASE_MODEL_NAME} and adapter from {LORA_ADAPTER_PATH}...")
    # diagnostics
    st.write("Checking filesystem for required files...")
    files_present = {}
    files_present['cwd'] = os.getcwd()
    files_present['repo_root_files'] = os.listdir(".")
    st.write(files_present)  # Shows files/dirs in repo root to help debugging

    # confirm adapter folder contents if exists
    adapter_exists = os.path.isdir(LORA_ADAPTER_PATH)
    adapter_files = []
    if adapter_exists:
        adapter_files = os.listdir(LORA_ADAPTER_PATH)
        st.write(f"Adapter folder contents: {adapter_files}")
    else:
        st.warning(f"Adapter folder not found at '{LORA_ADAPTER_PATH}'. Make sure you uploaded it to the repo root.")

    # check label classes file
    if not os.path.isfile(LE_CLASSES_FILE):
        st.error(f"Missing '{LE_CLASSES_FILE}' in repo root. Create it with a single line like: left,centre,right")
        return None, None, None

    # read label classes preserving order
    with open(LE_CLASSES_FILE, "r") as f:
        label_classes = [s.strip() for s in f.read().strip().split(",") if s.strip()]
    if len(label_classes) == 0:
        st.error(f"'{LE_CLASSES_FILE}' appears empty. Add comma separated labels in training order.")
        return None, None, None

    try:
        # Load base model
        st.write("Loading base model from Hugging Face (this may take a while)...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=len(label_classes)
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        # Load LoRA adapter (must contain adapter_config.json and weights)
        if adapter_exists:
            try:
                model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
                # ensure CPU
                device = torch.device("cpu")
                model.to(device)
            except Exception as e:
                st.error(f"Failed to load LoRA adapter. Error: {e}")
                st.warning("If adapter files are missing or named differently, PeftModel won't load. Check ft_lora_adapter contains adapter_config.json and adapter_model.bin (or pytorch_model.bin).")
                return None, None, None
        else:
            st.error("Adapter folder not found. App cannot run without the adapter weights.")
            return None, None, None

    except Exception as e:
        st.error(f"Base model loading failed: {e}")
        return None, None, None

    return model, tokenizer, label_classes

model, tokenizer, label_classes = load_assets()

st.set_page_config(page_title="LoRA Bias Detector", layout="wide")
st.title("📰 LoRA News Headline Bias Detector")
st.markdown("Model uses DistilBERT base downloaded at runtime, fine-tuned with a **tiny LoRA adapter** from GitHub.")
st.markdown("---")

headline_input = st.text_area(
    "Enter a News Headline to Analyze:",
    "The government's latest spending bill is a triumph for working families.",
    height=150
)

if model and tokenizer:
    if st.button("Analyze Bias", use_container_width=True, type="primary"):
        if not headline_input.strip():
            st.warning("Please enter a headline to analyze.")
        else:
            with st.spinner('Analyzing bias...'):
                device = torch.device("cpu")
                inputs = tokenizer(headline_input, return_tensors="pt", truncation=True, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                model.eval()
                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))

                # Map prediction index to label using label_classes (ensures correct mapping/order)
                try:
                    pred_label = label_classes[pred_idx]
                except Exception:
                    pred_label = f"idx_{pred_idx}"

                st.success("Analysis Complete!")
                st.subheader(f"Predicted Bias: **{pred_label}**")

                prob_df = pd.DataFrame({'Bias': label_classes, 'Probability': probs})
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                prob_df['Percentage'] = (prob_df['Probability'] * 100).round(1).astype(str) + '%'

                st.markdown("#### Probability Distribution")
                st.bar_chart(prob_df.set_index('Bias')['Probability'])
                st.table(prob_df[['Bias', 'Percentage']].rename(columns={'Percentage': 'Confidence'}))

else:
    st.error("Model assets failed to load. Check adapter folder and le_classes.txt are present and correctly named.")

