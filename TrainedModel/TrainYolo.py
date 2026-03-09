#To run this file cd into the TrainedModel folder and run: python TrainYolo.py
#To Train set TRAIN_MODEL to True and adjust TRAIN_EPOCHS as needed. 
from pathlib import Path
import importlib
import os

import kagglehub
import cv2
import torch
from ultralytics import YOLO


KAGGLE_DATASET = "doganozcan/traffic-sign-gtrb"
BASE_MODEL = "yolo11n.pt"
TEST_IMAGE = "giveWay.jpg"
BEST_WEIGHTS = "runs/detect/train/weights/best.pt" #Found in latest training run folder after doing 3 epochs

TRAIN_MODEL = False
RUN_TEST_AFTER_TRAIN = True

# Tweak Epoch for better accuracy
TRAIN_EPOCHS = 3
TRAIN_IMGSZ = 320
TRAIN_BATCH = -1  # Auto batch size for current hardware.
TRAIN_WORKERS = max(2, (os.cpu_count() or 4) - 2)
TRAIN_CACHE = "ram"


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def build_local_data_yaml(dataset_root: Path) -> Path:
    train_images = dataset_root / "train" / "images"
    val_images = dataset_root / "test" / "images"
    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(
            "Expected folders not found. Verify dataset layout contains train/images and test/images."
        )

    template = Path("data.yaml")
    lines = template.read_text(encoding="utf-8").splitlines()
    updated = []
    for line in lines:
        if line.startswith("train:"):
            updated.append(f"train: {train_images.as_posix()}")
        elif line.startswith("val:"):
            updated.append(f"val: {val_images.as_posix()}")
        else:
            updated.append(line)

    out_file = Path("data.local.yaml")
    out_file.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return out_file
def main() -> None:
    model = YOLO(BASE_MODEL)

    if TRAIN_MODEL:
        try:
            kagglehub = importlib.import_module("kagglehub")
        except ImportError as exc:
            raise ImportError(
                "kagglehub is required for dataset download. Install with: pip install kagglehub"
            ) from exc

        if KAGGLE_DATASET.startswith("YOUR_KAGGLE_USERNAME/"):
            raise ValueError(
                "Set KAGGLE_DATASET to your real Kaggle dataset id, e.g. 'owner/traffic-sign'."
            )

        print(f"Downloading dataset: {KAGGLE_DATASET}")
        dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        print(f"Dataset downloaded to: {dataset_path}")

        local_data_yaml = build_local_data_yaml(dataset_path)
        print(f"Training with config: {local_data_yaml}")

        model.train(
            data=str(local_data_yaml),
            epochs=TRAIN_EPOCHS,
            imgsz=TRAIN_IMGSZ,
            batch=TRAIN_BATCH,
            workers=TRAIN_WORKERS,
            device=resolve_device(),
            cache=TRAIN_CACHE,
            amp=True,
        )

    if RUN_TEST_AFTER_TRAIN:
        img = cv2.imread(TEST_IMAGE)
        if img is None:
            raise FileNotFoundError(f"Test image not found: {TEST_IMAGE}")

        best_model = YOLO(BEST_WEIGHTS)
        results = best_model(img)

        for r in results:
            if len(r.boxes) == 0:
                print("No objects detected in the image.")
            else:
                print(f"\nFound {len(r.boxes)} object(s):\n")
                for i, box in enumerate(r.boxes, 1):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = best_model.names[class_id]

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    width = x2 - x1
                    height = y2 - y1

                    print(f"{i}. {class_name.upper()}")
                    print(
                        f"   - Confidence: {confidence:.1%} (The model is {confidence:.1%} sure this is a {class_name})"
                    )
                    print(f"   - Size: {width:.0f}x{height:.0f} pixels")
                    print(f"   - Location: Center at ({(x1 + x2) / 2:.0f}, {(y1 + y2) / 2:.0f})")
                    print()

        results[0].show()


if __name__ == "__main__":
    main()