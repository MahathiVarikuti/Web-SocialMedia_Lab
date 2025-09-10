# Install required libraries if needed:
# pip install nltk textblob spacy spacytextblob
# python -m spacy download en_core_web_sm

import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from spacytextblob.spacytextblob import SpacyTextBlob

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load spaCy model and initialize SpacyTextBlob
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob", last=True)

# Sample product reviews
reviews = [
    "I absolutely love this phone, the camera is amazing!",
    "The battery life is terrible and it keeps overheating.",
    "It's okay, does the job but nothing special.",
    "Excellent build quality and very fast performance.",
    "Worst purchase ever! Completely useless after a week.",
    "Pretty decent for the price, but could be better."
]

# VADER Sentiment Analysis
print("=== VADER Sentiment Analysis ===")
vader = SentimentIntensityAnalyzer()
for review in reviews:
    scores = vader.polarity_scores(review)
    print(f"Review: {review}")
    print(f"Scores: {scores}")
    if scores['compound'] >= 0.05:
        print("Sentiment: Positive")
    elif scores['compound'] <= -0.05:
        print("Sentiment: Negative")
    else:
        print("Sentiment: Neutral")
    print("-" * 50)

# TextBlob Sentiment Analysis
print("=== TextBlob Sentiment Analysis ===")
for review in reviews:
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    print(f"Review: {review}")
    print(f"Polarity: {polarity}, Subjectivity: {subjectivity}")
    if polarity > 0:
        print("Sentiment: Positive")
    elif polarity < 0:
        print("Sentiment: Negative")
    else:
        print("Sentiment: Neutral")
    print("-" * 50)

# spaCy + SpacyTextBlob Sentiment Analysis
print("=== spaCy Sentiment Analysis ===")
for review in reviews:
    doc = nlp(review)
    print(f"Review: {review}")
    print(f"Polarity: {doc._.blob.polarity}, Subjectivity: {doc._.blob.subjectivity}")
    if doc._.blob.polarity > 0:
        print("Sentiment: Positive")
    elif doc._.blob.polarity < 0:
        print("Sentiment: Negative")
    else:
        print("Sentiment: Neutral")
    print("-" * 50)