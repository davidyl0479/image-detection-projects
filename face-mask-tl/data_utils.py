from __future__ import annotations

from pathlib import Path
import subprocess
import zipfile

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import xml.etree.ElementTree as ET
import random
from torchvision.models import ResNet18_Weights
import torchvision.transforms as T


def download_kaggle_dataset(
    dataset_slug: str = "andrewmvd/face-mask-detection",
    root_dir: str | Path = "data",
    force_download: bool = False,
) -> Path:
    """
    Download the Kaggle face mask dataset if it is not already present.

    Parameters
    ----------
    dataset_slug : str
        Kaggle dataset identifier in the form "owner/dataset-name".
    root_dir : str or Path
        Directory where the dataset folder will be created or found.
    force_download : bool
        If True, re-download and overwrite the existing dataset folder.

    Returns
    -------
    Path
        Path to the extracted dataset directory (containing images/ and annotations/).
    """
    root_dir = Path(root_dir)

    # We will name the dataset folder based on the slug, e.g.
    # "andrewmvd/face-mask-detection" -> "face-mask-detection"
    dataset_name = dataset_slug.split("/")[-1]
    dataset_dir = root_dir / dataset_name

    # If dataset already exists and we are not forcing a re-download, just return it
    if dataset_dir.exists() and not force_download:
        print(f"Using existing dataset at {dataset_dir}")
        return dataset_dir

    # Make sure the root directory exists
    root_dir.mkdir(parents=True, exist_ok=True)

    # Path to the zip file that Kaggle will download
    zip_path = root_dir / f"{dataset_name}.zip"

    print(f"Downloading {dataset_slug} to {zip_path} ...")
    # Call Kaggle CLI: kaggle datasets download -d <slug> -p <root_dir> -f <file>.zip -o
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(root_dir),
        "-o",  # overwrite if exists
    ]
    subprocess.run(cmd, check=True)

    # If Kaggle didn't name the file exactly <dataset_name>.zip, try to find any .zip in root_dir
    if not zip_path.exists():
        zip_files = list(root_dir.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError(
                f"No zip file found in {root_dir} after Kaggle download."
            )
        zip_path = zip_files[0]

    print(f"Extracting {zip_path} to {dataset_dir} ...")
    # Ensure target directory is clean if force_download is True
    if dataset_dir.exists() and force_download:
        # Be cautious: you could delete it here if you want a clean start
        # For now, we just reuse the folder and overwrite files during extraction.
        pass

    dataset_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)

    print(f"Dataset ready at {dataset_dir}")
    return dataset_dir


class FaceMaskDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of dicts, each with keys:
            - "image_path": Path
            - "bbox": (xmin, ymin, xmax, ymax)
            - "label_idx": int
        transform: torchvision transforms to apply to the cropped image
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # load image
        img = Image.open(sample["image_path"]).convert("RGB")

        # crop with bbox
        xmin, ymin, xmax, ymax = sample["bbox"]
        img = img.crop((xmin, ymin, xmax, ymax))

        # apply transforms
        if self.transform:
            img = self.transform(img)

        return img, sample["label_idx"]


def _build_samples_from_voc_xml(dataset_dir: Path) -> list[dict]:
    """
    Internal helper: parse all PASCAL VOC XML files and build a list of samples.

    Each sample is a dict with keys:
        - "image_path": Path
        - "bbox": (xmin, ymin, xmax, ymax)
        - "label_name": str
    """
    annotations_dir = dataset_dir / "annotations"
    images_dir = dataset_dir / "images"

    xml_files = sorted(annotations_dir.glob("*.xml"))

    samples: list[dict] = []

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text  # type: ignore
        image_path = images_dir / filename  # type: ignore

        for obj in root.findall("object"):
            class_name = obj.find("name").text  # type: ignore

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)  # type: ignore
            ymin = int(bndbox.find("ymin").text)  # type: ignore
            xmax = int(bndbox.find("xmax").text)  # type: ignore
            ymax = int(bndbox.find("ymax").text)  # type: ignore

            samples.append(
                {
                    "image_path": image_path,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "label_name": class_name,
                }
            )

    return samples


def create_dataloaders(
    dataset_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """
    Create train, validation and test DataLoaders for the face mask dataset.

    Parameters
    ----------
    dataset_dir : str or Path
        Path to the dataset directory containing images/ and annotations/.
    image_size : int
        Target size (shorter side) for the cropped face images.
    batch_size : int
        Number of samples per batch for all DataLoaders.
    val_ratio : float
        Fraction of the dataset to use for validation.
    test_ratio : float
        Fraction of the dataset to use for testing.
    num_workers : int
        Number of worker processes used by each DataLoader.
    seed : int
        Random seed used when shuffling and splitting the data.

    Returns
    -------
    (DataLoader, DataLoader, DataLoader, dict[str, int])
        Train, validation and test DataLoaders, and a mapping from class name
        to integer index.
    """
    dataset_dir = Path(dataset_dir)

    # 1. Parse XML annotations into a list of sample dicts
    samples = _build_samples_from_voc_xml(dataset_dir)

    if not samples:
        raise RuntimeError(f"No samples found in {dataset_dir}")

    # 2. Build class_to_idx mapping from label_name
    class_names = sorted({s["label_name"] for s in samples})
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # 3. Replace label_name with label_idx in each sample
    for s in samples:
        s["label_idx"] = class_to_idx[s["label_name"]]
        del s["label_name"]

    # 4. Shuffle and split into train / val / test
    random.seed(seed)
    random.shuffle(samples)

    n_total = len(samples)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test

    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    # 5. Define transforms using TorchVision weights (for ResNet18)
    weights = ResNet18_Weights.DEFAULT
    base_transforms = weights.transforms()

    # base_transforms already does resize + centre crop + to tensor + normalisation
    # For training we add some augmentation on top
    train_transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            base_transforms,
        ]
    )

    # For val/test we just use the base transforms (no augmentation)
    eval_transforms = base_transforms

    # 6. Create Dataset objects
    train_dataset = FaceMaskDataset(train_samples, transform=train_transforms)
    val_dataset = FaceMaskDataset(val_samples, transform=eval_transforms)
    test_dataset = FaceMaskDataset(test_samples, transform=eval_transforms)

    # 7. Wrap in DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_to_idx
