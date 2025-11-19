# app.py - Media Bias Detection (Sentiment-driven)
import streamlit as st
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Artifacts directory (should contain train_headlines.csv and train_embeddings.npy)
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "media_bias_artifacts")
TRAIN_HEADLINES_CSV = os.path.join(ARTIFACT_DIR, "train_headlines.csv")
TRAIN_EMB_NPY = os.path.join(ARTIFACT_DIR, "train_embeddings.npy")
SBERT_NAME = "all-MiniLM-L6-v2"

@st.cache_resource
def load_models():
    # Load SBERT and local artifacts if present; create fallbacks otherwise.
    sbert = SentenceTransformer(SBERT_NAME)
    if os.path.exists(TRAIN_HEADLINES_CSV) and os.path.exists(TRAIN_EMB_NPY):
        train_df = pd.read_csv(TRAIN_HEADLINES_CSV)
        train_emb = np.load(TRAIN_EMB_NPY)
    else:
        train_df = pd.DataFrame({"headline": [], "label": []})
        train_emb = np.zeros((0, sbert.get_sentence_embedding_dimension()))
    # Sentiment analysis pipeline (uses a small default model from HuggingFace)
    sentiment_model = pipeline("sentiment-analysis")
    return sbert, train_df, train_emb, sentiment_model

def classify_bias_knn(emb, train_emb, train_df, k=5):
    """Simple k-NN majority-vote classifier using cosine similarity on SBERT embeddings."""
    if train_emb.shape[0] == 0:
        return "unknown", []
    sims = cosine_similarity(emb, train_emb)[0]
    top_idx = np.argsort(-sims)[:k]
    nearest = [(int(i), train_df.loc[i, "label"], float(sims[i]), train_df.loc[i, "headline"]) for i in top_idx]
    labels = [n[1] for n in nearest]
    # majority vote; tie-breaker: highest-sum-similarity among tied labels
    from collections import Counter, defaultdict
    cnt = Counter(labels)
    top_count = max(cnt.values())
    candidates = [lab for lab, c in cnt.items() if c == top_count]
    if len(candidates) == 1:
        pred = candidates[0]
    else:
        # compute similarity sums for candidates
        sim_sum = defaultdict(float)
        for lab, sim in zip(labels, sims[top_idx] if hasattr(sims, '__getitem__') else [n[2] for n in nearest]):
            sim_sum[lab] += sim
        pred = max(candidates, key=lambda x: sim_sum[x])
    return pred, nearest

def main():
    st.set_page_config(page_title="Media Bias + Sentiment Detector", layout="wide")
    st.title("Media Bias Detection — Sentiment-driven (SBERT + kNN)")
    st.markdown(
        """
        This app replaces the previous Logistic Regression classifier with a simple **k-NN** approach over SBERT embeddings for political bias (left/center/right),
        and adds a **Transformer-based sentiment analysis** (positive/neutral/negative).
        """
    )

    sbert, train_df, train_emb, sentiment_model = load_models()

    st.sidebar.header("Settings")
    k = st.sidebar.slider("k (nearest neighbors) for bias classification", 1, 10, 5)
    top_k_display = st.sidebar.slider("Number of nearest examples to show", 1, 10, 4)

    st.subheader("Single headline prediction")
    user_headline = st.text_input("Enter a news headline", placeholder="Type or paste a headline...")
    if st.button("Predict headline"):
        if not user_headline.strip():
            st.warning("Please enter a headline.")
        else:
            with st.spinner("Computing..."):
                emb = sbert.encode([user_headline], convert_to_numpy=True)
                bias_label, nearest = classify_bias_knn(emb, train_emb, train_df, k=k)
                # sentiment analysis (transformers pipeline returns label and score)
                sent = sentiment_model(user_headline)[0]
            st.success(f"Predicted political bias: **{bias_label.upper()}**")
            st.info(f"Sentiment: **{sent['label']}**  (score={sent['score']:.3f})")

            # Show probabilities as rough proxy: compute normalized similarity sums per label among top 50
            if train_emb.shape[0] > 0:
                sims_all = cosine_similarity(emb, train_emb)[0]
                df_sim = pd.DataFrame({"label": train_df["label"].values, "sim": sims_all})
                sim_sum = df_sim.groupby("label")["sim"].sum()
                prob_proxy = (sim_sum / sim_sum.sum()).to_dict()
                prob_df = pd.Series(prob_proxy).reset_index()
                prob_df.columns = ["label", "probability"]
                st.write("Similarity-based probability proxy:")
                st.bar_chart(prob_df.set_index("label"))
            else:
                st.write("No training embeddings available to compute similarity-based probabilities.")

            st.markdown("**Nearest training examples (by cosine similarity):**")
            for idx, lab, sim, h in nearest[:top_k_display]:
                st.write(f"- `{h}` — **{lab}** (sim={sim:.3f})")

    st.markdown("---")
    st.subheader("Batch prediction (CSV)")
    uploaded = st.file_uploader("Upload CSV with a `headline` column for batch prediction", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "headline" not in df.columns:
            st.error("CSV must contain a `headline` column.")
        else:
            st.info(f"Processing {len(df)} rows...")
            embs = sbert.encode(df['headline'].astype(str).tolist(), convert_to_numpy=True, show_progress_bar=True)
            preds = []
            sents = []
            for i, e in enumerate(embs):
                pred_label, _ = classify_bias_knn([e], train_emb, train_df, k=k)
                sentiment = sentiment_model(df.loc[i, "headline"])[0]
                preds.append(pred_label)
                sents.append(sentiment["label"])
            out = df.copy()
            out["predicted_bias"] = preds
            out["sentiment"] = sents
            st.dataframe(out.head(200))
            csv = out.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", csv, "predictions_sentiment_bias.csv", "text/csv")

    st.markdown("---")
    st.caption("Modeling approach: SBERT embeddings used for semantic similarity. Bias classification is a nearest-neighbor (k-NN) majority-vote over training headlines. Sentiment uses a HuggingFace transformer pipeline.")

if __name__ == "__main__":
    main()
