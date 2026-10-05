import os
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import pipeline

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Sentiment Analysis API",
    description="API для анализа тональности текста на базе предобученной модели RuBERT",
    version="1.0.0"
)

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "sentiment-analysis",
    model="blanchefort/rubert-base-cased-sentiment",
    device=device
)

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: TextRequest):
    result = classifier(request.text)[0]
    return {
        "label": result["label"],
        "score": round(float(result["score"]), 4)
    }