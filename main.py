# pip install kagglehub ultralytics

from pathlib import Path
import shutil
import random
import kagglehub
from ultralytics import YOLO


# =========================================================
# CONFIG
# =========================================================
KAGGLE_HANDLE = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
RAW_DIR_NAME = "gtsrb_raw"
YOLO_DIR_NAME = "gtsrb_yolo"

MODEL_NAME = "yolo11n-cls.pt"
EPOCHS = 10
IMGSZ = 64
BATCH = 64
DEVICE = "cpu"   # use "0" if you want GPU
VAL_SPLIT = 0.2
SEED = 42


def find_image_root(download_path: Path) -> Path:
    """
    Find the actual dataset root.

    We want either:
    1) a directory containing split folders like Train/Test/Meta
    or
    2) a directory containing class folders directly
    """
    split_names = {"train", "test", "val", "valid", "meta"}

    # Search all directories, shallow to deep
    candidates = [download_path] + [p for p in download_path.rglob("*") if p.is_dir()]

    for candidate in candidates:
        children = [p for p in candidate.iterdir() if p.is_dir()]
        if not children:
            continue

        child_names = {c.name.lower() for c in children}

        # Case 1: dataset root contains split folders
        if child_names & split_names:
            return candidate

        # Case 2: dataset root contains class folders directly
        numeric_dirs = [c for c in children if c.name.isdigit()]
        if len(numeric_dirs) >= 10:
            return candidate

    raise FileNotFoundError("Could not find dataset root with splits or class folders.")


def copy_images(src_dir: Path, dst_dir: Path, exts: set[str]):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img in src_dir.iterdir():
        if img.is_file() and img.suffix.lower() in exts:
            shutil.copy2(img, dst_dir / img.name)


def build_yolo_classification_dataset(src_root: Path, out_root: Path, val_split: float = 0.2, seed: int = 42):
    """
    Builds:
        out_root/
            train/class_x/
            val/class_x/
            test/class_x/

    Handles:
    - existing train/test split folders
    - class folders directly under root
    """
    random.seed(seed)
    exts = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}

    if out_root.exists():
        shutil.rmtree(out_root)

    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "test").mkdir(parents=True, exist_ok=True)

    children = [p for p in src_root.iterdir() if p.is_dir()]
    name_map = {p.name.lower(): p for p in children}

    # -------------------------------------------------
    # Case 1: source already has split folders
    # -------------------------------------------------
    train_src = None
    test_src = None

    for k, v in name_map.items():
        if k == "train":
            train_src = v
        elif k == "test":
            test_src = v

    if train_src is not None:
        # Build train/val from source train folder
        class_dirs = [p for p in train_src.iterdir() if p.is_dir()]

        for class_dir in class_dirs:
            images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
            if not images:
                continue

            random.shuffle(images)
            n_val = max(1, int(len(images) * val_split)) if len(images) > 1 else 0

            val_imgs = images[:n_val]
            train_imgs = images[n_val:]

            train_class_dir = out_root / "train" / class_dir.name
            val_class_dir = out_root / "val" / class_dir.name

            train_class_dir.mkdir(parents=True, exist_ok=True)
            val_class_dir.mkdir(parents=True, exist_ok=True)

            for img in train_imgs:
                shutil.copy2(img, train_class_dir / img.name)

            for img in val_imgs:
                shutil.copy2(img, val_class_dir / img.name)

        # If source test folder already has class folders, copy them
        if test_src is not None:
            test_class_dirs = [p for p in test_src.iterdir() if p.is_dir()]
            for class_dir in test_class_dirs:
                copy_images(class_dir, out_root / "test" / class_dir.name, exts)

        return

    # -------------------------------------------------
    # Case 2: source has class folders directly
    # -------------------------------------------------
    class_dirs = [p for p in children if p.name.isdigit() or p.is_dir()]
    if not class_dirs:
        raise FileNotFoundError("No class directories found in source root.")

    for class_dir in class_dirs:
        images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not images:
            continue

        random.shuffle(images)
        n_val = max(1, int(len(images) * val_split)) if len(images) > 1 else 0

        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        train_class_dir = out_root / "train" / class_dir.name
        val_class_dir = out_root / "val" / class_dir.name

        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(img, train_class_dir / img.name)

        for img in val_imgs:
            shutil.copy2(img, val_class_dir / img.name)

def print_split_counts(dataset_root: Path):
    for split in ["train", "val", "test"]:
        split_path = dataset_root / split
        if not split_path.exists():
            print(f"{split}: missing")
            continue

        total = 0
        classes = 0
        for class_dir in split_path.iterdir():
            if class_dir.is_dir():
                classes += 1
                total += len([p for p in class_dir.iterdir() if p.is_file()])
        print(f"{split}: {classes} classes, {total} images")


def train_baseline(data_root: str):
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=data_root,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="runs/gtsrb_baseline",
        name="yolo11n_cls_baseline"
    )

    print("\nTraining finished.")
    print(results)


def evaluate(model_path: str, data_root: str, split: str):
    model = YOLO(model_path)
    metrics = model.val(
        data=data_root,
        split=split,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="runs/gtsrb_baseline",
        name=f"eval_{split}"
    )
    print(f"\n{split.upper()} METRICS")
    print(metrics)


def main():
    # Download full dataset files to local cache/disk
    download_path = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    print("Downloaded to:", download_path)

    # Optional copy to a named local folder
    raw_dir = Path(RAW_DIR_NAME)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    shutil.copytree(download_path, raw_dir)

    # Find where the class folders actually live
    src_root = find_image_root(raw_dir)
    print("Detected image root:", src_root)

    # Convert into YOLO classification folder structure
    yolo_root = Path(YOLO_DIR_NAME)
    build_yolo_classification_dataset(src_root, yolo_root, val_split=VAL_SPLIT, seed=SEED)

    print("\nBuilt YOLO dataset at:", yolo_root)
    print_split_counts(yolo_root)

    # Train baseline
    train_baseline(str(yolo_root))

    # Evaluate
    best_model = Path("runs/gtsrb_baseline/yolo11n_cls_baseline/weights/best.pt")
    if best_model.exists():
        evaluate(str(best_model), str(yolo_root), "val")

        # Only run test if test actually has images
        test_dir = yolo_root / "test"
        has_test_images = any(
            class_dir.is_dir() and any(p.is_file() for p in class_dir.iterdir())
            for class_dir in test_dir.iterdir()
        )
        if has_test_images:
            evaluate(str(best_model), str(yolo_root), "test")
        else:
            print("\nNo populated test split was found, so test evaluation was skipped.")
    else:
        print("\nCould not find best.pt after training.")


if __name__ == "__main__":
    main()