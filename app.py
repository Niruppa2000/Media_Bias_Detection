# app.py - Media Bias Detection (Headlines)
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set ARTIFACT_DIR relative to app location or environment
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "media_bias_artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "logreg_sbert_clf.joblib")
LE_PATH = os.path.join(ARTIFACT_DIR, "label_encoder.joblib")
TRAIN_HEADLINES_CSV = os.path.join(ARTIFACT_DIR, "train_headlines.csv")
TRAIN_EMB_NPY = os.path.join(ARTIFACT_DIR, "train_embeddings.npy")
SBERT_NAME = "all-MiniLM-L6-v2"  # SBERT model name (downloaded at runtime)

@st.cache_resource
def load_models():
    # loads classifier and label encoder from artifacts and loads SBERT model
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        st.warning("Classifier or label encoder not found in artifacts. The app may not behave as expected.")
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    sbert = SentenceTransformer(SBERT_NAME)
    # load training headlines and embeddings
    if os.path.exists(TRAIN_HEADLINES_CSV) and os.path.exists(TRAIN_EMB_NPY):
        train_h = pd.read_csv(TRAIN_HEADLINES_CSV)
        train_emb = np.load(TRAIN_EMB_NPY)
    else:
        train_h = pd.DataFrame({"headline": [], "label": []})
        train_emb = np.zeros((0, sbert.get_sentence_embedding_dimension()))
    return clf, le, sbert, train_h, train_emb

def predict(headline: str, clf, le, sbert):
    h = headline.strip()
    emb = sbert.encode([h], convert_to_numpy=True)
    probs = clf.predict_proba(emb)[0]
    pred_idx = int(probs.argmax())
    pred_label = le.inverse_transform([pred_idx])[0]
    prob_map = {le.inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))}
    return pred_label, prob_map, emb

def nearest_examples(query_emb, train_emb, train_df, top_k=3):
    if train_emb.shape[0] == 0:
        return []
    sims = cosine_similarity(query_emb, train_emb)[0]
    top_idx = np.argsort(-sims)[:top_k]
    rows = []
    for i in top_idx:
        rows.append({"headline": train_df.loc[i, "headline"], "label": train_df.loc[i, "label"], "similarity": float(sims[i])})
    return rows

def main():
    st.set_page_config(page_title="Media Bias Detector (Headlines)", layout="wide")
    st.title("Media Bias Detection — Headline Classifier")
    st.markdown(
        """
        Classify news headlines as **left**, **center**, or **right** leaning using SBERT embeddings + Logistic Regression.
        - Enter a headline to get prediction + probability distribution.
        - Upload a CSV (`headline` column) for batch predictions.
        """
    )

    clf, le, sbert, train_df, train_emb = load_models()

    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of nearest examples to show", 1, 10, 3)

    st.subheader("Single headline prediction")
    user_headline = st.text_input("Enter a news headline", "")
    if st.button("Predict headline"):
        if not user_headline.strip():
            st.warning("Please enter a headline.")
        else:
            with st.spinner("Computing prediction..."):
                pred_label, prob_map, emb = predict(user_headline, clf, le, sbert)
            st.success(f"Predicted bias: **{pred_label.upper()}**")
            st.write("Probabilities:")
            prob_df = pd.Series(prob_map).reset_index()
            prob_df.columns = ["label", "probability"]
            st.bar_chart(data=prob_df.set_index("label"))

            st.markdown("**Nearest training examples (by cosine similarity):**")
            nearest = nearest_examples(emb, train_emb, train_df, top_k=top_k)
            for ex in nearest:
                st.write(f"- `{ex['headline']}`  —  **{ex['label']}**  (sim={ex['similarity']:.3f})")

    st.markdown("---")
    st.subheader("Batch prediction (CSV)")
    uploaded = st.file_uploader("Upload CSV with a `headline` column", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "headline" not in df.columns:
            st.error("CSV must contain a `headline` column.")
        else:
            st.info(f"Predicting {len(df)} rows")
            emb_batch = sbert.encode(df['headline'].astype(str).tolist(), convert_to_numpy=True, show_progress_bar=True)
            probs = clf.predict_proba(emb_batch)
            pred_idx = probs.argmax(axis=1)
            preds = le.inverse_transform(pred_idx)
            out = df.copy()
            out["predicted_label"] = preds
            for i, cls in enumerate(le.classes_):
                out[f"prob_{cls}"] = probs[:, i]
            st.dataframe(out.head(200))
            csv = out.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")

    st.markdown("---")
    st.caption("Model: SBERT (all-MiniLM-L6-v2) embeddings + LogisticRegression. This repo includes synthetic artifacts for a quick demo. For real results replace artifacts with model trained on SBERT embeddings from a labeled dataset.")

if __name__ == "__main__":
    main()
