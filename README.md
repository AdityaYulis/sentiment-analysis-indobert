# 🇮🇩 Sentiment Analysis with IndoBERT + Streamlit

A Deep Learning project for **Indonesian sentiment analysis** using
**IndoBERT**, deployed as an interactive web application with
**Streamlit**.

This project classifies text reviews into **Neutral** , **Positive** or **Negative**
sentiment using a fine-tuned transformer-based model.

------------------------------------------------------------------------

## 🚀 Demo

Run the application locally:

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📌 Features

-   🔍 Sentiment classification (Neutral, Positive / Negative)
-   🤖 Powered by IndoBERT transformer model
-   🧠 Fine-tuned for Indonesian text data
-   💬 User input for real-time prediction
-   🌐 Simple and interactive Streamlit UI

------------------------------------------------------------------------

## 🧠 Model Overview

-   Pretrained model: IndoBERT\
-   Architecture: Transformer-based text classification\
-   Task: Sentiment Analysis\
-   Output:
    -   Neutral 😐
    -   Positive 😊
    -   Negative 😠

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
git clone https://github.com/AdityaYulis/sentiment-analysis-indobert.git
cd sentiment-analysis-indobert
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Run App

``` bash
streamlit run app.py
```

Open: http://localhost:8501

------------------------------------------------------------------------

## 📊 Example

  Input                            Sentiment
  -------------------------------- -----------
  "Biasa saja"                     Neutral
  "Aplikasi ini sangat membantu"   Positive
  "Pelayanan buruk sekali"         Negative

------------------------------------------------------------------------

## 🛠 Tech Stack

Python, IndoBERT, PyTorch, Transformers, Streamlit, Scikit-learn

------------------------------------------------------------------------

## 👨‍💻 Author

Aditya Yulis

------------------------------------------------------------------------
