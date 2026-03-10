# TrainYolo.py
# Examples:
#   python TrainYolo.py --train
#   python TrainYolo.py --test
#   python TrainYolo.py --train --test
#   python TrainYolo.py --score
#   python TrainYolo.py --train --score
#   python TrainYolo.py --train --epochs 10 --dataset "archive/Traffic Signs"
#   python TrainYolo.py --test --image giveWay.jpg --weights runs/detect/train/weights/best.pt
from utils import (
    resolve_device,
    find_latest_best_weights,
    build_local_data_yaml,
    save_confusion_matrix_each_epoch,
    validate_dataset_class_ids,
)
from pathlib import Path
from ultralytics import YOLO
import argparse
import os
import cv2

BASE_MODEL = "yolo11n.pt"
DEFAULT_DATASET_ROOT = "./archive/Traffic Signs"
DEFAULT_TEST_IMAGE = "giveWay.jpg"


def train_model(args) -> Path | None:
    dataset_root = Path(args.dataset.strip())

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    print(f"Using dataset at: {dataset_root.resolve()}")

    # Fail fast on dataset metadata mismatches instead of silently skipping labels.
    validate_dataset_class_ids(dataset_root)

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


def score_model(args, weights_override: Path | None = None) -> None:
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

    dataset_root = Path(args.dataset.strip())
    local_data_yaml = build_local_data_yaml(dataset_root)

    print(f"\nScoring model: {weights_path.resolve()}")
    print(f"Validation data: {local_data_yaml.resolve()}\n")

    model = YOLO(str(weights_path))
    metrics = model.val(data=str(local_data_yaml), device=resolve_device(), plots=True)

    mp = metrics.box.mp
    mr = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    class_names = model.names  # {id: name}
    ap50_per_class = metrics.box.ap50      # array aligned to ap_class_index
    ap_per_class = metrics.box.ap          # mAP50-95 per class
    class_indices = metrics.box.ap_class_index

    lines = []
    lines.append("=" * 64)
    lines.append("  SCORING REPORT")
    lines.append("=" * 64)
    lines.append(f"  Weights  : {weights_path}")
    lines.append(f"  Data     : {local_data_yaml}")
    lines.append("-" * 64)
    lines.append(f"  {'Metric':<30} {'Value':>10}")
    lines.append("-" * 64)
    lines.append(f"  {'Precision (mean)':<30} {mp:>10.4f}")
    lines.append(f"  {'Recall (mean)':<30} {mr:>10.4f}")
    lines.append(f"  {'mAP@0.50':<30} {map50:>10.4f}")
    lines.append(f"  {'mAP@0.50:0.95':<30} {map5095:>10.4f}")
    lines.append("=" * 64)
    lines.append(f"  {'Class':<30} {'AP50':>8}  {'AP50-95':>8}")
    lines.append("-" * 64)
    for idx, cls_id in enumerate(class_indices):
        name = class_names.get(int(cls_id), str(cls_id))
        lines.append(f"  {name:<30} {ap50_per_class[idx]:>8.4f}  {ap_per_class[idx]:>8.4f}")
    lines.append("=" * 64)

    report = "\n".join(lines)
    print(report)

    report_path = weights_path.parent.parent / "scoring_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


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
    parser.add_argument("--score", action="store_true", help="Run validation scoring report")

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
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--cache", type=str, default="disk", help="Dataset cache mode")

    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for Ultralytics outputs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name inside the project directory (default: auto-incremented by Ultralytics)",
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

    if not args.train and not args.test and not args.score:
        parser.error("You must pass at least one flag: --train, --test, and/or --score")

    return args


def main():
    args = parse_args()

    trained_best = None

    if args.train:
        trained_best = train_model(args)

    if args.score:
        score_model(args, weights_override=trained_best)

    if args.test:
        test_model(args, weights_override=trained_best)


if __name__ == "__main__":
    main()