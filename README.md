# Прикладные решения на основе готовых библиотек машинного обучения

Студент: Исаев Всеволод 11-411

## Стек и задачи

В проекте решены 4 прикладные задачи с использованием 3 различных ML-фреймворков:

1. **Обработка текста (NLP):** Анализ тональности текста (Sentiment Analysis).
   * **Библиотека:** Hugging Face (`transformers`, PyTorch).
   * **Модель:** `blanchefort/rubert-base-cased-sentiment`.
2. **Обработка аудио:** Распознавание речи в текст (Speech-to-Text).
   * **Библиотека:** Hugging Face (`transformers`, PyTorch).
   * **Модель:** `openai/whisper-tiny`.
3. **Обработка изображений:** Классификация объектов на фото.
   * **Библиотека:** TensorFlow / Keras Applications.
   * **Модель:** `MobileNetV2` (веса `imagenet`).
4. **Обработка видео:** Детекция объектов в видеопотоке.
   * **Библиотека:** PyTorch (`ultralytics`).
   * **Модель:** `YOLOv8n` (веса `coco`).
5. **Сервис API и автоматизация:**
   * **REST API:** FastAPI + Uvicorn (эндпоинт `/predict` для анализа тональности).
   * **Тестирование:** PyTest + FastAPI TestClient (HTTPX).
   * **CI/CD:** GitHub Actions (автоматический запуск тестов при `push` и `pull_request`).

## Быстрый старт

1. Клонирование репозитория:
   ```bash
   git clone [https://github.com/](https://github.com/)<ваш-логин>/SE-Isaev-Vsevolod-11-411.git
   cd SE-Isaev-Vsevolod-11-411
   ```
   
2. Создание и активация виртуального окружения (Python 3.11)
    ```bash
   python -m venv .venv

    # Для Windows (PowerShell):
    .\.venv\Scripts\Activate.ps1
    
    # Для Windows (CMD):
    .\.venv\Scripts\activate.bat
    
    # Для Linux / macOS:
    source .venv/bin/activate
   ```
   
3. Установка зависимостей
    ```bash
   pip install -r requirements.txt
   ```
   
4. Запуск решений
    ```bash
    python task1_text/predict.py
    python task2_audio/transcribe.py
    python task3_image/classify.py
    python task4_video/detect.py
   ```
   
### Запуск REST API (FastAPI)
   ```bash
   uvicorn main:app --reload
   ```
После старта сервера доступны:
   * Адрес сервиса: http://127.0.0.1:8000
   * Интерактивная документация (Swagger UI): http://127.0.0.1:8000/docs
   * Альтернативная документация (ReDoc): http://127.0.0.1:8000/redoc

### Тестирование и CI/CD
   ```bash
   pytest -v
   ```