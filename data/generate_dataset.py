"""
generate_dataset.py
--------------------
Creates a synthetic (template-based) sentiment analysis dataset with
2,000+ labeled samples across 3 classes: Positive, Neutral, Negative.

WHY A SYNTHETIC DATASET?
The project spec (PDF) lists dataset options (Twitter/X API, Kaggle
Amazon/Yelp reviews, Facebook Graph API, internal feedback forms) but
does not pin down one exact source or link. Some of those require
internet access / API keys / manual Kaggle downloads that this
environment cannot do for you automatically.

This script generates a realistic, template-based dataset of product/
service feedback sentences so the ENTIRE pipeline (train -> evaluate ->
Streamlit app) works immediately, out of the box, with no downloads.

TO USE A REAL DATASET INSTEAD (recommended for a real portfolio project):
1. Download a dataset such as:
   - Kaggle: "Amazon Product Reviews" or "Twitter Sentiment140"
   - Any CSV with a text column and a sentiment/rating column
2. Convert it into a CSV with EXACTLY two columns: text,label
   - label must be one of: Positive, Neutral, Negative
   - If your source uses star ratings, map them like:
     1-2 stars -> Negative, 3 stars -> Neutral, 4-5 stars -> Positive
3. Save it as data/dataset.csv (overwriting the generated one) and re-run
   train.py

Run this script with:  python data/generate_dataset.py
"""

import random
import csv
import os

random.seed(42)  # reproducible dataset

# ---------------------------------------------------------------------------
# Template building blocks for each sentiment class.
# Sentences are built by combining subjects + phrases to create variety.
# ---------------------------------------------------------------------------

SUBJECTS = [
    "The product", "This app", "Your service", "The delivery", "The support team",
    "This laptop", "The food", "The hotel room", "The course", "This website",
    "The customer service", "The new update", "This phone", "The event",
    "My experience", "The staff", "The tool", "This feature", "The internship program",
    "The training session", "The software", "This subscription", "The packaging",
    "The instructor", "The response time", "The interface", "This gadget", "The meeting",
]

POSITIVE_PHRASES = [
    "exceeded my expectations and I loved every part of it",
    "was absolutely fantastic, I would highly recommend it to anyone",
    "worked perfectly and made my day so much easier",
    "is amazing, the quality is outstanding",
    "made me really happy, thank you so much",
    "was a wonderful experience from start to finish",
    "is excellent and works exactly as promised",
    "impressed me a lot, great job by the team",
    "was super fast and extremely helpful",
    "is fantastic, I can't stop recommending it",
    "was smooth, friendly, and very professional",
    "gave me great value for the price",
    "is the best I have used so far, truly brilliant",
    "was delightful and exceeded all my needs",
    "is very reliable and I am thoroughly satisfied",
    "made everything so simple and enjoyable",
    "was well organized and genuinely useful",
    "is a great improvement over the last version",
    "left me feeling very satisfied and impressed",
    "worked flawlessly, no complaints at all",
]

NEUTRAL_PHRASES = [
    "was okay, nothing special but not bad either",
    "is average, it does what it says",
    "was fine, I don't have strong feelings about it",
    "is acceptable for the price, nothing more nothing less",
    "worked as expected, no surprises",
    "was a normal experience overall",
    "is fairly standard compared to other options",
    "did the job but felt a bit generic",
    "was neither good nor bad, just okay",
    "is decent, could be improved in a few areas",
    "met the basic requirements, nothing outstanding",
    "was moderate in quality and service",
    "is functional but not particularly exciting",
    "seemed reasonable, I have mixed feelings",
    "was an average experience, could go either way",
    "is fine for casual use, nothing extraordinary",
    "worked most of the time without major issues",
    "was okay overall, I might use it again",
    "is a middle-of-the-road option",
    "left me with a neutral impression",
]

NEGATIVE_PHRASES = [
    "was a complete disappointment and a waste of money",
    "did not work at all, very frustrating experience",
    "is terrible, I would not recommend it to anyone",
    "broke down after just one use, poor quality",
    "was extremely slow and unhelpful",
    "made me really upset, I want a refund",
    "is the worst experience I have had in a long time",
    "was rude and unprofessional, very disappointing",
    "failed to deliver what was promised",
    "is buggy and crashes constantly",
    "was a horrible waste of time",
    "left me feeling completely dissatisfied",
    "is overpriced for such poor quality",
    "was confusing and poorly designed",
    "did not meet my expectations at all",
    "is unreliable and keeps causing problems",
    "was a disaster from start to finish",
    "is disappointing compared to what was advertised",
    "made the whole process painful and stressful",
    "was awful, I regret using it",
]

# Extra noisy variants to mimic real-world text (urls, html, punctuation, caps)
NOISE_SUFFIXES = [
    "", "", "", "  Check it out at https://example.com/review",
    " <br> Posted from my phone.", "!!!", "...", " #feedback", " @support",
    " Visit http://example.org for details.", "",
]


def build_sentence(subject: str, phrase: str) -> str:
    """Combine a subject and phrase into a full sentence, adding light noise."""
    sentence = f"{subject} {phrase}."
    suffix = random.choice(NOISE_SUFFIXES)
    if suffix:
        sentence = f"{sentence}{suffix}"
    # Randomly uppercase the whole sentence sometimes (mimics shouting reviews)
    if random.random() < 0.05:
        sentence = sentence.upper()
    return sentence


def generate_samples(n_per_class: int):
    """Generate n_per_class samples for each of the 3 sentiment classes."""
    rows = []
    class_phrase_map = {
        "Positive": POSITIVE_PHRASES,
        "Neutral": NEUTRAL_PHRASES,
        "Negative": NEGATIVE_PHRASES,
    }
    for label, phrases in class_phrase_map.items():
        for _ in range(n_per_class):
            subject = random.choice(SUBJECTS)
            phrase = random.choice(phrases)
            text = build_sentence(subject, phrase)
            rows.append((text, label))
    return rows


def main():
    samples_per_class = 700  # 700 x 3 = 2,100 total samples (>= 2,000 required)
    rows = generate_samples(samples_per_class)
    random.shuffle(rows)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "dataset.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Dataset created: {output_path}")
    print(f"Total samples: {len(rows)}")
    print("Class distribution:")
    for label in ["Positive", "Neutral", "Negative"]:
        count = sum(1 for _, l in rows if l == label)
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
