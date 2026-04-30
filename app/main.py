from fastapi import FastAPI

from app.predictor import predict_sentiment

app = FastAPI()

@app.get("/")
def root():

    return {
        "message":
        "IndoBERT Review API Running"
    }


@app.post("/predict")

def predict(review: str):

    sentiment = predict_sentiment(
        review
    )

    return {

        "review": review,
        "sentiment": sentiment

    }