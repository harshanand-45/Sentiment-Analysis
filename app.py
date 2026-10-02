"""
app.py
------
Streamlit web application for the Sentiment Analysis project.

Features:
- Paste/type any text and click Predict -> get Positive/Neutral/Negative
- Shows confidence/probability for each class as a bar chart
- Colour-coded result: Green = Positive, Grey = Neutral, Red = Negative
- Batch prediction: upload a CSV, get sentiment + confidence for every row,
  download the results as a new CSV

Run with:  streamlit run app.py
"""

import os
import pickle

import pandas as pd
import streamlit as st

from src.preprocessing import clean_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

SENTIMENT_COLORS = {
    "Positive": "#2ecc71",  # green
    "Neutral": "#95a5a6",   # grey
    "Negative": "#e74c3c",  # red
}

REQUIRED_CSV_COLUMN = "text"  # column name expected in uploaded CSV for batch mode


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    """
    Load the trained model and TF-IDF vectorizer from disk.
    Returns (model, vectorizer) or (None, None) if files are missing.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_sentiment(text: str, model, vectorizer):
    """
    Run the full prediction pipeline on a single piece of text:
    clean -> vectorize -> predict -> return label + probability dict.

    Returns:
        (predicted_label, {class_name: probability, ...})
        or (None, None) if the input text is empty after cleaning.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return None, None

    features = vectorizer.transform([cleaned])
    predicted_label = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    class_names = model.classes_
    prob_dict = {cls: prob for cls, prob in zip(class_names, probabilities)}

    return predicted_label, prob_dict


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis Tool")
st.write(
    "Classify any text feedback as **Positive**, **Neutral**, or **Negative** "
    "using a Logistic Regression model trained on TF-IDF features."
)

model, vectorizer = load_model_and_vectorizer()

if model is None or vectorizer is None:
    st.error(
        "⚠️ Model files not found.\n\n"
        "Please train the model first by running:\n\n"
        "```\npython train.py\n```\n\n"
        "This will create the required files in the `models/` folder:\n"
        "- `models/sentiment_model.pkl`\n"
        "- `models/tfidf_vectorizer.pkl`"
    )
    st.stop()

tab1, tab2 = st.tabs(["📝 Single Text Prediction", "📂 Batch CSV Prediction"])

# ---------------------------------------------------------------------------
# TAB 1: Single text prediction
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Enter text to analyze")
    user_text = st.text_area(
        "Paste or type feedback / review / comment here:",
        height=150,
        placeholder="e.g. The product was amazing and delivery was super fast!",
    )

    if st.button("Predict Sentiment", type="primary"):
        if not user_text or not user_text.strip():
            st.warning("⚠️ Please enter some text before clicking Predict.")
        else:
            label, probs = predict_sentiment(user_text, model, vectorizer)

            if label is None:
                st.warning(
                    "⚠️ After cleaning, no meaningful words were left in this text "
                    "(e.g. it may contain only links, numbers, or symbols). "
                    "Please try a different input."
                )
            else:
                color = SENTIMENT_COLORS.get(label, "#3498db")
                st.markdown(
                    f"""
                    <div style="padding: 20px; border-radius: 10px;
                                background-color: {color}22; border: 2px solid {color};
                                text-align: center;">
                        <h2 style="color: {color}; margin: 0;">{label}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("### Confidence / Probability Breakdown")
                prob_df = pd.DataFrame(
                    {"Sentiment": list(probs.keys()), "Probability": list(probs.values())}
                ).sort_values("Sentiment")
                st.bar_chart(prob_df.set_index("Sentiment"))

                # Also show exact percentages
                cols = st.columns(len(probs))
                for col, (cls, prob) in zip(cols, sorted(probs.items())):
                    col.metric(cls, f"{prob * 100:.1f}%")

# ---------------------------------------------------------------------------
# TAB 2: Batch CSV prediction
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Upload a CSV file for batch prediction")
    st.write(
        f"Your CSV **must** contain a column named exactly **`{REQUIRED_CSV_COLUMN}`** "
        "containing the text to classify. Any other columns will be kept as-is."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"⚠️ Could not read this CSV file. Error: {e}")
            batch_df = None

        if batch_df is not None:
            if REQUIRED_CSV_COLUMN not in batch_df.columns:
                st.error(
                    f"⚠️ Missing required column '{REQUIRED_CSV_COLUMN}'. "
                    f"Found columns: {list(batch_df.columns)}"
                )
            elif batch_df.empty:
                st.warning("⚠️ The uploaded CSV file is empty.")
            else:
                with st.spinner("Running predictions on all rows..."):
                    predicted_labels = []
                    confidences = []

                    for text_value in batch_df[REQUIRED_CSV_COLUMN]:
                        label, probs = predict_sentiment(str(text_value), model, vectorizer)
                        if label is None:
                            predicted_labels.append("Unknown")
                            confidences.append(0.0)
                        else:
                            predicted_labels.append(label)
                            confidences.append(round(max(probs.values()) * 100, 2))

                    batch_df["predicted_sentiment"] = predicted_labels
                    batch_df["confidence_percent"] = confidences

                st.success(f"✅ Predictions complete for {len(batch_df)} rows.")
                st.dataframe(batch_df, use_container_width=True)

                # Summary chart of sentiment distribution in the batch
                st.write("### Sentiment Distribution")
                st.bar_chart(batch_df["predicted_sentiment"].value_counts())

                # Download button
                csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_bytes,
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                )

st.markdown("---")
st.caption(
    "Model: Logistic Regression + TF-IDF | "
    "Trained on a labeled dataset with 3 classes: Positive, Neutral, Negative."
)
