import pandas as pd

from datasets import Dataset

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

def load_dataset():

    df = pd.read_csv(
        "data/processed/processed_reviews.csv"
    )

    df["label"] = df["sentiment"].map(
        label_map
    )

    dataset = Dataset.from_pandas(df)

    return dataset