from google_play_scraper import reviews
import pandas as pd

APP_ID = "com.kai.kaiticketing"
TOTAL_REVIEWS = 5000

all_reviews = []

print("Scraping reviews...")

result, continuation_token = reviews(
    APP_ID,
    lang="id",
    country="id",
    count=200
)

all_reviews.extend(result)

while len(all_reviews) < TOTAL_REVIEWS:

    result, continuation_token = reviews(
        APP_ID,
        continuation_token=continuation_token
    )

    if not result:
        break

    all_reviews.extend(result)

print("Total reviews:", len(all_reviews))

df = pd.DataFrame(all_reviews)

df = df[[
    "content",
    "score"
]]

df.columns = [
    "review",
    "rating"
]

df.to_csv(
    "data/raw/raw_reviews.csv",
    index=False
)

print("Saved raw data.")