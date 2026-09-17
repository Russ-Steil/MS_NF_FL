"""
PyTorch datasets for Federated Learning.

Two paths:
  - UniversalOCTDataset reads images with ImageFolder and does crop/resize on
    the fly. Kept for reference and for uncached runs.
  - CachedOCTDataset reads the uint8 tensors written by cache_images.py, where
    crop and resize have already been applied.

The cached transforms reproduce the on-the-fly ones exactly: uint8 -> float in
[0,1] is what ToTensor did, and rotation, flip, jitter and normalization all
operate on tensors identically.
"""
from pathlib import Path
import json
import os
from typing import Optional, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets


# ---------------------------------------------------------------- on the fly

def get_train_transform(image_size: int = 224, crop_img = True):
    return transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=400, height=img.height, width=img.width - 400)) if crop_img else transforms.Lambda(lambda img: img),
        transforms.Resize((image_size, 2*image_size)), 
        transforms.ToTensor(), 
        transforms.RandomRotation(15), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

def get_val_transform(image_size: int = 224, crop_img = True):
    return transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=400, height=img.height, width=img.width - 400)) if crop_img else transforms.Lambda(lambda img: img),
        transforms.Resize((image_size, 2*image_size)),  
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class UniversalOCTDataset(datasets.ImageFolder):
    def __init__(self, img_dir: str, image_size: int = 224, transform=None, **kwargs):
        super().__init__(root=img_dir, transform=transform)
        print(f"Dataset caricato: {len(self)} immagini, classi: {self.class_to_idx}")

UniPD_Dataset = UniversalOCTDataset
UCD_Dataset = UniversalOCTDataset


# ---------------------------------------------------------------- cached

def _load_cached_tensor(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def get_cached_train_transform():
    """Crop and resize are already baked into the cache."""
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),   # uint8 -> [0,1], as ToTensor did
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_cached_val_transform():
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CachedOCTDataset(datasets.DatasetFolder):
    """Reads .pt tensors written by cache_images.py. Exposes .targets like ImageFolder."""

    def __init__(self, img_dir: str, transform=None, **kwargs):
        super().__init__(
            root=img_dir,
            loader=_load_cached_tensor,
            extensions=(".pt",),
            transform=transform,
        )
        print(f"Cached dataset loaded: {len(self)} tensors, classes: {self.class_to_idx}")


def read_cache_manifest(cache_path):
    """Returns the manifest dict, or raises if the cache is missing or incomplete."""
    path = Path(cache_path) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no manifest.json in {cache_path} — run cache_images.py first"
        )
    with open(path) as f:
        return json.load(f)
