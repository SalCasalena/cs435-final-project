import torch
import shutil
from pathlib import Path
import re

def resolve_device() -> str:
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def _extract_nc_from_yaml(yaml_text: str) -> int | None:
    match = re.search(r"^\s*nc\s*:\s*(\d+)\s*$", yaml_text, flags=re.MULTILINE)
    if match:
        return int(match.group(1))
    return None


def _max_class_id_in_labels(labels_root: Path) -> int | None:
    max_id: int | None = None
    for label_file in labels_root.glob("*.txt"):
        for raw_line in label_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            token = line.split()[0]
            try:
                class_id = int(float(token))
            except ValueError:
                continue
            if max_id is None or class_id > max_id:
                max_id = class_id
    return max_id


def validate_dataset_class_ids(dataset_root: Path) -> None:
    """
    Validate that labels under train/labels do not exceed nc from dataset YAML.
    Raises a ValueError with a clear message when classes are mismatched.
    """
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        return

    yaml_text = data_yaml.read_text(encoding="utf-8")
    nc = _extract_nc_from_yaml(yaml_text)
    if nc is None:
        return

    labels_root = dataset_root / "train" / "labels"
    if not labels_root.exists():
        return

    max_label_id = _max_class_id_in_labels(labels_root)
    if max_label_id is None:
        return

    if max_label_id >= nc:
        raise ValueError(
            f"Dataset class mismatch: max class id in labels is {max_label_id}, "
            f"but nc is {nc} in {data_yaml}. "
            f"Expected valid class ids: 0-{nc - 1}."
        )

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
    test_images = dataset_root / "test" / "images"
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

    template = dataset_root / "data.yaml"
    if not template.exists():
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
        elif line.startswith("test:") and test_images.exists():
            updated.append(f"test: {test_images.resolve().as_posix()}")
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
