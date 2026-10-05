import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def classify_image(image_path: str, top_k: int = 3):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл {image_path} не найден!")

    model = MobileNetV2(weights="imagenet")

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)

    decoded_results = decode_predictions(predictions, top=top_k)[0]
    return decoded_results


if __name__ == "__main__":
    image_file = os.path.join(CURRENT_DIR, "sample.jpg")

    print("Классификация изображений (Keras MobileNetV2)")
    print(f"Обработка файла: {image_file}...\n")

    results = classify_image(image_file, top_k=3)

    print("Топ предсказаний:")
    for rank, (class_id, class_name, probability) in enumerate(results, start=1):
        print(f"{rank}. {class_name} ({probability * 100:.2f}%)")