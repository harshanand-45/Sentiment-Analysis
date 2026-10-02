# 5-Minute Demo / Presentation Script

**Total time: ~5 minutes.** Have `streamlit run app.py` already running in a
terminal before you start.

## 1. Introduction (30 seconds)
> "I built a Sentiment Analysis tool that classifies any feedback text as
> Positive, Neutral, or Negative. It uses NLTK for text preprocessing,
> TF-IDF for feature extraction, and a Logistic Regression model — a
> lightweight but effective approach for text classification."

## 2. Pipeline walkthrough (1 minute)
> "The pipeline has 5 stages: dataset → preprocessing → TF-IDF vectorization
> → Logistic Regression training → evaluation. Preprocessing lowercases the
> text, strips URLs/HTML/special characters, tokenizes it, removes
> stopwords, and lemmatizes each word — for example 'running' becomes 'run'.
> The exact same cleaning function is used at prediction time, so the model
> always sees text prepared consistently."

(Optionally show `src/preprocessing.py`'s `clean_text` function briefly.)

## 3. Show evaluation results (1 minute)
> "After training, the model is evaluated on a held-out test set using
> accuracy, precision, recall, and F1-score, plus a confusion matrix."

Open `models/confusion_matrix.png` and `models/evaluation_report.json`, or
just show the printed output from `train.py` in the terminal.

## 4. Live demo — single prediction (1.5 minutes)
Switch to the Streamlit app, **Single Text Prediction** tab. Enter these
three examples one at a time, clicking Predict each time:

1. `"This app exceeded my expectations, I loved every part of it!"` → expect **Positive** (green)
2. `"The product was okay, nothing special but not bad either."` → expect **Neutral** (grey)
3. `"This was a complete disappointment and a waste of money."` → expect **Negative** (red)

> "Notice the colour coding — green for positive, grey for neutral, red for
> negative — and the probability bar chart showing the model's confidence
> across all three classes."

## 5. Live demo — batch CSV prediction (30 seconds)
Switch to the **Batch CSV Prediction** tab. Upload a small CSV with a `text`
column containing a few example sentences.

> "The app can also process a whole CSV of feedback at once, adding a
> predicted sentiment and confidence score to every row, and letting you
> download the results."

## 6. Key learnings (30 seconds)
> "Key things I learned: how to build a full ML pipeline end-to-end — data,
> preprocessing, feature extraction, training, and evaluation; the
> importance of keeping preprocessing identical between training and
> inference; and how to turn a trained model into a usable web app with
> Streamlit."

## 7. Close (10 seconds)
> "That's the full sentiment analysis tool — happy to answer any questions."
