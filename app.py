import os
import numpy as np
import pandas as pd
import torch
import streamlit as st

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL_NAME = "distilbert-base-uncased"
LORA_ADAPTER_PATH = "ft_lora_adapter"          # folder exported from Colab
HEAD_FILE = os.path.join(LORA_ADAPTER_PATH, "cls_head.pt")
LE_CLASSES_FILE = "le_classes.txt"             # e.g. "Left,Neutral,Right"


# =========================
# LOAD MODEL + TOKENIZER + LABELS
# =========================
@st.cache_resource
def load_assets():
    """Load base model, LoRA adapter, classifier head, tokenizer, and labels."""
    # ---- Load label classes ----
    if not os.path.exists(LE_CLASSES_FILE):
        st.error(
            f"'{LE_CLASSES_FILE}' not found in repo root. "
            "Make sure you uploaded it from Colab."
        )
        return None, None, None

    try:
        with open(LE_CLASSES_FILE, "r") as f:
            label_classes = [
                s.strip()
                for s in f.read().strip().split(",")
                if s.strip()
            ]
    except Exception as e:
        st.error(f"Error reading '{LE_CLASSES_FILE}': {e}")
        return None, None, None

    if len(label_classes) == 0:
        st.error(f"'{LE_CLASSES_FILE}' appears empty. Add comma-separated labels.")
        return None, None, None

    # ---- Load base DistilBERT classifier ----
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=len(label_classes),
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    except Exception as e:
        st.error(f"Error loading base model/tokenizer: {e}")
        return None, None, None

    # ---- Attach LoRA adapter ----
    if not os.path.isdir(LORA_ADAPTER_PATH):
        st.error(
            f"Adapter folder '{LORA_ADAPTER_PATH}' not found in repo root. "
            "Upload the 'ft_lora_adapter' folder from Colab."
        )
        return None, None, None

    try:
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    except Exception as e:
        st.error(f"Error loading LoRA adapter from '{LORA_ADAPTER_PATH}': {e}")
        return None, None, None

    # ---- Load fine-tuned classifier head ----
    if os.path.exists(HEAD_FILE):
        try:
            head_state = torch.load(HEAD_FILE, map_location="cpu")
            model.base_model.pre_classifier.load_state_dict(head_state["pre_classifier"])
            model.base_model.classifier.load_state_dict(head_state["classifier"])
            st.success("Loaded fine-tuned classifier head (cls_head.pt).")
        except Exception as e:
            st.error(f"Error loading classifier head from '{HEAD_FILE}': {e}")
            st.warning("Using RANDOM classifier head; predictions may be poor.")
    else:
        st.warning(
            f"Classifier head file '{HEAD_FILE}' not found. "
            "Using RANDOM classifier head; model may behave badly."
        )

    # ---- Move to CPU, eval mode ----
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, label_classes


model, tokenizer, label_classes = load_assets()


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="LoRA News Bias Detector", layout="wide")

st.title("📰 LoRA Political Bias Detector")
st.markdown(
    """
This app uses **DistilBERT** with a **LoRA adapter** and a fine-tuned classifier head  
to predict whether a news headline is **Left-leaning**, **Right-leaning**, or **Neutral/Centre**.
"""
)
st.markdown("---")

headline_input = st.text_area(
    "Enter a news headline:",
    "Government announces new climate policy to reduce emissions over the next decade.",
    height=150,
)

if model and tokenizer:
    if st.button("Analyze Bias", use_container_width=True, type="primary"):
        if not headline_input.strip():
            st.warning("Please enter a headline to analyze.")
        else:
            with st.spinner("Analyzing bias..."):
                device = torch.device("cpu")
                inputs = tokenizer(
                    headline_input,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                pred_idx = int(np.argmax(probs))
                pred_label = label_classes[pred_idx]

                st.success("Analysis complete!")
                st.subheader(f"Predicted Bias: **{pred_label}**")

                # Probability distribution
                prob_df = pd.DataFrame(
                    {
                        "Bias": label_classes,
                        "Probability": probs,
                    }
                )
                prob_df = prob_df.sort_values("Probability", ascending=False)
                prob_df["Confidence"] = (prob_df["Probability"] * 100).round(1).astype(str) + "%"

                st.markdown("### Probability Distribution")
                st.bar_chart(prob_df.set_index("Bias")["Probability"])
                st.table(prob_df[["Bias", "Confidence"]])
else:
    st.error(
        "Model assets failed to load. "
        "Check that 'ft_lora_adapter', 'cls_head.pt', and 'le_classes.txt' "
        "are present in your GitHub repository root."
    )
