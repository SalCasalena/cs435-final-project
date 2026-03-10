import torch
import shutil
from pathlib import Path

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
