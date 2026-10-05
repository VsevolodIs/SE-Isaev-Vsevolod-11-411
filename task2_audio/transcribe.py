import os
import numpy as np
import soundfile as sf
from scipy.signal import resample
import torch
from transformers import pipeline


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SAMPLE_RATE = 16000

def transcribe_audio(audio_path: str):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Файл не найден: {audio_path}")

    audio_data, sample_rate = sf.read(audio_path)

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        num_target_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / sample_rate)
        audio_data = resample(audio_data, num_target_samples)
        sample_rate = TARGET_SAMPLE_RATE

    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-tiny",
        device=device
    )

    result = pipe({"raw": audio_data.astype(np.float32), "sampling_rate": sample_rate})
    return result["text"]

if __name__ == "__main__":
    audio_file = os.path.join(CURRENT_DIR, "sample.wav")

    print("=== Распознавание речи (Whisper-tiny) ===")
    print(f"Обработка файла: {audio_file}...")

    text = transcribe_audio(audio_file)
    print("\nРезультат распознавания:")
    print(text)