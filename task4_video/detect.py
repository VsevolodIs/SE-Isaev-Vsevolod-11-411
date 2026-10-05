import os
import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def process_video(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Файл {video_path} не найден!")

    print(f"Загрузка модели YOLOv8n и обработка видео: {video_path}...")

    model = YOLO("yolov8n.pt")

    output_dir = os.path.join(CURRENT_DIR, "runs")

    results = model.predict(
        source=video_path,
        save=True,
        project=output_dir,
        name="detection_output",
        exist_ok=True,
        conf=0.35
    )

    save_path = os.path.join(output_dir, "detection_output")
    print(f"\nГотово! Обработанный видеоролик сохранён в папку:\n{save_path}")


if __name__ == "__main__":
    video_file = os.path.join(CURRENT_DIR, "sample.gif")

    print("=== Детекция объектов на видео (YOLOv8 / PyTorch) ===")
    process_video(video_file)