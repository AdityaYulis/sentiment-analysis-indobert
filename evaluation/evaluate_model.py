import pandas as pd

from sklearn.metrics import classification_report

from app.predictor import predict_sentiment

df = pd.read_csv(
    "data/processed/processed_reviews.csv"
)

y_true = df["sentiment"]

y_pred = []

for text in df["review"]:

    pred = predict_sentiment(text)

    y_pred.append(pred)

print(
    classification_report(
        y_true,
        y_pred
    )
)