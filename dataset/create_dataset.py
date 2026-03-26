"""
dataset/create_dataset.py

Downloads the Massachusetts Roads Dataset from Kaggle and organises it into:
    dataset/
        images/   ← aerial RGB tiles (.tiff)
        masks/    ← binary road masks (.tif)
        splits/   ← train.txt  val.txt  test.txt  (stem names, one per line)

Requirements:
    pip install kaggle tqdm

Kaggle credentials:
    Place kaggle.json in ~/.kaggle/kaggle.json (chmod 600 on Linux/Mac)
    or set KAGGLE_USERNAME and KAGGLE_KEY env vars.
    See https://github.com/Kaggle/kaggle-api#api-credentials

Usage:
    python dataset/create_dataset.py
    python dataset/create_dataset.py --train 0.946 --val 0.012 --seed 42
"""

import argparse
import os
import random
import shutil
import zipfile
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
IMAGES_DIR   = SCRIPT_DIR / "images"
MASKS_DIR    = SCRIPT_DIR / "masks"
SPLITS_DIR   = SCRIPT_DIR / "splits"
DOWNLOAD_DIR = SCRIPT_DIR / "_download"

KAGGLE_DATASET = "balraj98/massachusetts-roads-dataset"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_kaggle(dest: Path) -> None:
    try:
        import kaggle  # noqa: F401 — triggers credential check
    except ImportError:
        raise SystemExit("Install the Kaggle API:  pip install kaggle")

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} → {dest}")
    os.system(f'kaggle datasets download -d {KAGGLE_DATASET} -p "{dest}" --unzip')


def find_files(root: Path, suffixes: tuple) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in suffixes)


def copy_files(files: list[Path], dest: Path, desc: str) -> list[str]:
    """Copy files to dest, return list of stems."""
    dest.mkdir(parents=True, exist_ok=True)
    stems = []
    for src in tqdm(files, desc=desc):
        shutil.copy2(src, dest / src.name)
        stems.append(src.stem)
    return stems


def write_split(path: Path, stems: list[str]) -> None:
    path.write_text("\n".join(stems) + "\n")
    print(f"  {path.name}: {len(stems)} samples")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(train_frac: float, val_frac: float, seed: int) -> None:
    assert abs(train_frac + val_frac - 1.0) > 1e-6 or val_frac > 0, \
        "train + val must be < 1.0 so test gets the remainder"
    test_frac = 1.0 - train_frac - val_frac
    assert test_frac >= 0, "train_frac + val_frac must be ≤ 1.0"

    # 1. Download ----------------------------------------------------------------
    if not DOWNLOAD_DIR.exists() or not any(DOWNLOAD_DIR.rglob("*.tiff")):
        download_kaggle(DOWNLOAD_DIR)
    else:
        print(f"Skipping download — files already in {DOWNLOAD_DIR}")

    # 2. Locate images and masks -------------------------------------------------
    image_files = find_files(DOWNLOAD_DIR, (".tiff", ".tif", ".png", ".jpg"))
    # The Massachusetts Roads dataset names masks with the same stem as images;
    # masks live in a subfolder whose name contains "mask" or "label".
    mask_files  = [f for f in image_files
                   if any(part in f.parts for part in ("mask", "masks", "label", "labels"))
                   or "_mask" in f.stem or "_label" in f.stem]
    image_files = [f for f in image_files if f not in set(mask_files)]

    # Match by stem so images[i] ↔ masks[i]
    mask_by_stem = {f.stem.replace("_mask", "").replace("_label", ""): f
                    for f in mask_files}
    paired = [(img, mask_by_stem[img.stem])
              for img in image_files if img.stem in mask_by_stem]

    if not paired:
        raise RuntimeError(
            f"Could not pair images with masks under {DOWNLOAD_DIR}.\n"
            f"Found {len(image_files)} image(s) and {len(mask_files)} mask(s).\n"
            "Check the downloaded folder structure and adjust the pairing logic."
        )

    print(f"\nPaired {len(paired)} image/mask samples")

    # 3. Copy to organised folders -----------------------------------------------
    imgs, msks = zip(*paired)
    img_stems = copy_files(list(imgs), IMAGES_DIR, "Copying images")
    _          = copy_files(list(msks), MASKS_DIR,  "Copying masks")

    # 4. Create splits -----------------------------------------------------------
    random.seed(seed)
    indices = list(range(len(img_stems)))
    random.shuffle(indices)

    n_train = int(len(indices) * train_frac)
    n_val   = int(len(indices) * val_frac)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    stems = [img_stems[i] for i in range(len(img_stems))]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nSplits:")
    write_split(SPLITS_DIR / "train.txt", [stems[i] for i in train_idx])
    write_split(SPLITS_DIR / "val.txt",   [stems[i] for i in val_idx])
    write_split(SPLITS_DIR / "test.txt",  [stems[i] for i in test_idx])

    print(f"\nDone. Dataset ready in {SCRIPT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Massachusetts Roads dataset layout")
    parser.add_argument("--train", type=float, default=0.946,
                        help="Fraction for training   (default: 0.946 → ~1108/1171)")
    parser.add_argument("--val",   type=float, default=0.012,
                        help="Fraction for validation (default: 0.012 → ~14/1171)")
    parser.add_argument("--seed",  type=int,   default=42,
                        help="Random seed for split shuffle (default: 42)")
    args = parser.parse_args()
    main(args.train, args.val, args.seed)
