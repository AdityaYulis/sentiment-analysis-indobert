import pandas as pd

from sklearn.model_selection import train_test_split

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

df = pd.read_csv(
    "data/processed/processed_reviews.csv"
)

df = df.dropna(subset=["clean_review"])

df["clean_review"] = df["clean_review"].astype(str)

df = df[df["clean_review"].str.strip() != ""]

print("Dataset size after cleaning:", len(df))

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["sentiment"].map(
    label_map
)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

train_dataset = Dataset.from_pandas(
    train_df
)

test_dataset = Dataset.from_pandas(
    test_df
)

tokenizer = AutoTokenizer.from_pretrained(
    "indobenchmark/indobert-base-p1"
)

def tokenize(batch):

    return tokenizer(
        batch["clean_review"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

print("Tokenizing dataset...")

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p1",
    num_labels=3
)

training_args = TrainingArguments(
    output_dir="model/indobert",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Training model...")

trainer.train()

trainer.save_model(
    "model/indobert"
)

tokenizer.save_pretrained(
    "model/indobert"
)

print("IndoBERT model saved.")