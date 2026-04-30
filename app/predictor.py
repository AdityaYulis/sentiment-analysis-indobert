import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from utils.preprocess import clean_text

tokenizer = AutoTokenizer.from_pretrained(
    "model/indobert"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "model/indobert"
)

labels = [
    "negative",
    "neutral",
    "positive"
]

def predict_sentiment(text):

    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits

    predicted_class = torch.argmax(
        logits,
        dim=1
    ).item()

    return labels[predicted_class]