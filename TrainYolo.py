# TrainYolo.py
# Examples:
#   python TrainYolo.py --train
#   python TrainYolo.py --test
#   python TrainYolo.py --train --test
#   python TrainYolo.py --train --epochs 10 --dataset "archive/Traffic Signs"
#   python TrainYolo.py --test --image giveWay.jpg --weights runs/detect/train/weights/best.pt

from pathlib import Path
import argparse
import os
import shutil

import cv2
import torch
from ultralytics import YOLO


BASE_MODEL = "yolo11n.pt"
DEFAULT_DATASET_ROOT = "./archive/Traffic Signs"
DEFAULT_TEST_IMAGE = "giveWay.jpg"


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def find_latest_best_weights() -> Path | None:
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        return None

    candidates = list(runs_dir.glob("*/weights/best.pt"))
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_local_data_yaml(dataset_root: Path) -> Path:
    train_images = dataset_root / "train" / "images"
    val_images = dataset_root / "valid" / "images"
    train_labels = dataset_root / "train" / "labels"
    val_labels = dataset_root / "valid" / "labels"

    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(
            f"Expected folders not found under {dataset_root}:\n"
            f"  - {train_images}\n"
            f"  - {val_images}"
        )

    if not train_labels.exists() or not val_labels.exists():
        raise FileNotFoundError(
            f"Expected folders not found under {dataset_root}:\n"
            f"  - {train_labels}\n"
            f"  - {val_labels}"
        )

    template = Path("data.yaml")
    if not template.exists():
        raise FileNotFoundError("Could not find data.yaml in the current folder.")

    lines = template.read_text(encoding="utf-8").splitlines()
    updated = []

    for line in lines:
        if line.startswith("train:"):
            updated.append(f"train: {train_images.resolve().as_posix()}")
        elif line.startswith("val:"):
            updated.append(f"val: {val_images.resolve().as_posix()}")
        else:
            updated.append(line)

    out_file = Path("data.local.yaml")
    out_file.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return out_file


def save_confusion_matrix_each_epoch(trainer) -> None:
    """
    Copies Ultralytics-generated confusion matrix images into a per-epoch folder.

    This assumes validation is running and Ultralytics has written one or both of:
      - confusion_matrix.png
      - confusion_matrix_normalized.png
    into the run directory by the end of the epoch.
    """
    epoch_num = trainer.epoch + 1
    save_dir = Path(trainer.save_dir)

    epoch_dir = save_dir / "epoch_confusion_matrices"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        save_dir / "confusion_matrix.png",
        save_dir / "confusion_matrix_normalized.png",
    ]

    copied_any = False
    for src in candidates:
        if src.exists():
            dst = epoch_dir / f"epoch_{epoch_num:03d}_{src.name}"
            shutil.copy2(src, dst)
            print(f"Saved {dst}")
            copied_any = True

    if not copied_any:
        print(f"No confusion matrix image found to copy after epoch {epoch_num}.")


def train_model(args) -> Path | None:
    dataset_root = Path(args.dataset.strip())

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    print(f"Using dataset at: {dataset_root.resolve()}")

    local_data_yaml = build_local_data_yaml(dataset_root)
    print(f"Training with config: {local_data_yaml.resolve()}")

    model = YOLO(args.base_model)
    model.add_callback("on_fit_epoch_end", save_confusion_matrix_each_epoch)

    results = model.train(
        data=str(local_data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=resolve_device(),
        cache=args.cache,
        amp=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    best_path = None
    if hasattr(results, "save_dir"):
        run_dir = Path(results.save_dir)
        candidate = run_dir / "weights" / "best.pt"
        if candidate.exists():
            best_path = candidate

    if best_path is None:
        best_path = find_latest_best_weights()

    if best_path is not None:
        print(f"Best weights: {best_path}")
    else:
        print("Could not automatically find best.pt after training.")

    return best_path


def test_model(args, weights_override: Path | None = None) -> None:
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    if weights_override is not None:
        weights_path = weights_override
    elif args.weights:
        weights_path = Path(args.weights)
    else:
        latest = find_latest_best_weights()
        if latest is None:
            raise FileNotFoundError(
                "No weights provided and no best.pt found in runs/detect."
            )
        weights_path = latest

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"Testing image: {image_path.resolve()}")
    print(f"Using weights: {weights_path.resolve()}")

    model = YOLO(str(weights_path))
    results = model(str(image_path))

    for r in results:
        if len(r.boxes) == 0:
            print("No objects detected in the image.")
        else:
            print(f"\nFound {len(r.boxes)} object(s):\n")
            for i, box in enumerate(r.boxes, 1):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                print(f"{i}. {class_name.upper()}")
                print(
                    f"   - Confidence: {confidence:.1%} "
                    f"(The model is {confidence:.1%} sure this is a {class_name})"
                )
                print(f"   - Size: {width:.0f}x{height:.0f} pixels")
                print(f"   - Location: Center at ({center_x:.0f}, {center_y:.0f})")
                print()

    annotated = results[0].plot()
    cv2.imshow("YOLO Prediction", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or test a YOLO traffic sign detector."
    )

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model on an image")

    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Path to local dataset root containing train/ and valid/",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL,
        help="Base YOLO model to fine-tune",
    )

    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=320, help="Training image size")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 = auto)")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(2, (os.cpu_count() or 4) - 2),
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="ram",
        help="Dataset cache mode (e.g. ram, disk, False)",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for Ultralytics outputs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Run name inside the project directory",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_TEST_IMAGE,
        help="Path to test image",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained weights (.pt). If omitted, the newest best.pt is used.",
    )

    args = parser.parse_args()

    if not args.train and not args.test:
        parser.error("You must pass at least one flag: --train and/or --test")

    return args


def main():
    args = parse_args()

    trained_best = None

    if args.train:
        trained_best = train_model(args)

    if args.test:
        test_model(args, weights_override=trained_best)


if __name__ == "__main__":
    main()