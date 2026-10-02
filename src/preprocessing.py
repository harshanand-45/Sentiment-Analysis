"""
preprocessing.py
-----------------
Reusable text preprocessing pipeline using NLTK.

IMPORTANT: This exact function (clean_text) is used during BOTH
training (train.py) and prediction (app.py) so that the model always
sees text processed the same way. If you change anything here, you
must retrain the model afterwards.

Pipeline steps (as required by the project spec):
1. Lowercase the text
2. Remove URLs
3. Remove HTML tags
4. Remove special characters / punctuation / numbers
5. Tokenize
6. Remove stopwords
7. Lemmatize
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_resources():
    """
    Make sure all required NLTK resources are available.
    Downloads them quietly if missing. Safe to call multiple times.
    """
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


# Download resources once when this module is imported.
ensure_nltk_resources()

# Initialize reusable objects once (faster than recreating per call).
_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Precompiled regex patterns for speed and clarity.
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_PATTERN = re.compile(r"<.*?>")
_SPECIAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Apply the full preprocessing pipeline to a single piece of text and
    return a cleaned string ready for TF-IDF vectorization.

    Args:
        text: Raw input text (may be None or empty).

    Returns:
        Cleaned, lemmatized text as a single space-separated string.
        Returns an empty string if input is empty/invalid.
    """
    if text is None:
        return ""

    text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = _URL_PATTERN.sub(" ", text)

    # 3. Remove HTML tags
    text = _HTML_PATTERN.sub(" ", text)

    # 4. Remove special characters / punctuation / digits (keep letters + spaces)
    text = _SPECIAL_CHARS_PATTERN.sub(" ", text)

    # Collapse multiple spaces into one and strip
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()

    if not text:
        return ""

    # 5. Tokenization
    tokens = word_tokenize(text)

    # 6. Stop-word removal (and drop very short leftover tokens)
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    # 7. Lemmatization
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


if __name__ == "__main__":
    # Quick manual test when running this file directly:
    #   python src/preprocessing.py
    sample = "I REALLY loved this product!!! Visit https://example.com <br> It's amazing :)"
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
