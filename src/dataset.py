"""
dataset.py — DataLoader, transforms, and augmentation for Massachusetts Roads Dataset.

Reads split files (train.txt, val.txt, test.txt) containing one stem per line,
then loads the corresponding image (.tiff) and mask (.tif) from the dataset directory.
"""

import os
import random

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class MassRoadsDataset(Dataset):
    """
    Massachusetts Roads Dataset — binary road segmentation.

    Each sample is a (image, mask) pair where:
      image : FloatTensor [3, H, W]  — normalized RGB satellite image
      mask  : FloatTensor [1, H, W]  — binary road mask (1 = road, 0 = background)
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        stems: list[str],
        images_dir: Path,
        masks_dir: Path,
        img_size: int = 512,
        augment: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        image_ext: str = ".tiff",
        mask_ext: str = ".tif",
    ):
        self.stems        = stems
        self.images_dir   = images_dir
        self.masks_dir    = masks_dir
        self.img_size     = img_size
        self.augment      = augment
        self.mean         = mean or self.MEAN
        self.std          = std or self.STD
        self.normalize    = T.Normalize(mean=self.mean, std=self.std)
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
        self.image_ext    = image_ext
        self.mask_ext     = mask_ext

    def __len__(self) -> int:
        return len(self.stems)

    def _load_pair(self, idx: int):
        stem  = self.stems[idx]
        img   = Image.open(self.images_dir / f"{stem}{self.image_ext}").convert("RGB")
        label = Image.open(self.masks_dir  / f"{stem}{self.mask_ext}").convert("L")
        return img, label

    def _apply_transforms(self, img: Image.Image, label: Image.Image):
        img   = TF.resize(img,   [self.img_size, self.img_size],
                          interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        label = TF.resize(label, [self.img_size, self.img_size],
                          interpolation=T.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img, label = TF.hflip(img), TF.hflip(label)
            if random.random() > 0.5:
                img, label = TF.vflip(img), TF.vflip(label)
            angle = random.choice([0, 90, 180, 270])
            if angle:
                img   = TF.rotate(img,   angle)
                label = TF.rotate(label, angle)
            img = self.color_jitter(img)

        img   = TF.to_tensor(img)
        label = TF.to_tensor(label)
        label = (label > 0.5).float()
        img   = self.normalize(img)
        return img, label

    def __getitem__(self, idx: int):
        img, label = self._load_pair(idx)
        return self._apply_transforms(img, label)

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(MassRoadsDataset.MEAN).view(3, 1, 1)
        std  = torch.tensor(MassRoadsDataset.STD).view(3, 1, 1)
        return (tensor * std + mean).clamp(0, 1)


def _read_stems(path: Path) -> list[str]:
    """Read a split file (one stem per line) and return list of stems."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(
    dataset_root: Path,
    img_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 0 if os.name == "nt" else 4,
    seed: int = 42,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.1,
    image_ext: str = ".tiff",
    mask_ext: str = ".tif",
):
    """Build train / val / test DataLoaders from split files."""
    images_dir = dataset_root / "images"
    masks_dir  = dataset_root / "masks"
    splits_dir = dataset_root / "splits"

    train_stems = _read_stems(splits_dir / "train.txt")
    val_stems   = _read_stems(splits_dir / "val.txt")
    test_stems  = _read_stems(splits_dir / "test.txt")

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = dict(images_dir=images_dir, masks_dir=masks_dir, img_size=img_size,
                  mean=mean, std=std, brightness=brightness, contrast=contrast,
                  saturation=saturation, image_ext=image_ext, mask_ext=mask_ext)

    train_ds = MassRoadsDataset(train_stems, augment=True,  **common)
    val_ds   = MassRoadsDataset(val_stems,   augment=False, **common)
    test_ds  = MassRoadsDataset(test_stems,  augment=False, **common)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
