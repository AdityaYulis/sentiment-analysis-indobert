import pandas as pd

from utils.preprocess import clean_text

print("Loading raw data...")

df = pd.read_csv(
    "data/raw/raw_reviews.csv"
)

print("Cleaning text...")

df["clean_review"] = df["review"].apply(
    clean_text
)

def label_sentiment(rating):

    if rating >= 4:
        return "positive"

    elif rating == 3:
        return "neutral"

    else:
        return "negative"


df["sentiment"] = df["rating"].apply(
    label_sentiment
)

df.to_csv(
    "data/processed/processed_reviews.csv",
    index=False
)

print("Saved processed data.")