import torch
from transformers import pipeline

def analyze_sentiment(text: str):
    classifier = pipeline(
        "sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment"
    )
    results = classifier(text)
    return results

if __name__ == "__main__":
    sample_texts = [
        "Мне очень понравился этот курс, материал подан великолепно!",
        "Сервис ужасный, приложение постоянно вылетает и ничего не работает.",
        "Завтра в два часа дня состоится запланированная лекция."
    ]

    print("Результаты анализа тональности")
    for text in sample_texts:
        result = analyze_sentiment(text)[0]
        label = result["label"]
        score = round(result["score"], 4)
        print(f"\nТекст: {text}")
        print(f"Тональность: {label} (Уверенность: {score * 100:.2f}%)")