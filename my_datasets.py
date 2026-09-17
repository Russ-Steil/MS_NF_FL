"""PyTorch datasets for Federated Learning.

Two paths:
  - UniversalOCTDataset reads images with ImageFolder and does crop/resize on
    the fly. Kept for reference and for uncached runs.
  - CachedOCTDataset reads the uint8 tensors written by cache_images.py, where
    crop and resize have already been applied.

The cached transforms reproduce the on-the-fly ones exactly: uint8 -> float in
[0,1] is what ToTensor did, and rotation, flip, jitter and normalization all
operate on tensors identically.
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import re
from typing import Callable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------- on the fly


def get_train_transform(image_size: int = 224, crop_img=True):
  return transforms.Compose([
      (
          transforms.Lambda(
              lambda img: transforms.functional.crop(
                  img, top=0, left=500, height=img.height, width=img.width - 500
              )
          )
          if crop_img
          else transforms.Lambda(lambda img: img)
      ),
      transforms.Resize((image_size, 2 * image_size)),
      transforms.ToTensor(),
      transforms.RandomRotation(15),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ColorJitter(
          brightness=0.1, contrast=0.1, saturation=0.1
      ),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])


def get_val_transform(image_size: int = 224, crop_img=True):
  return transforms.Compose([
      (
          transforms.Lambda(
              lambda img: transforms.functional.crop(
                  img, top=0, left=500, height=img.height, width=img.width - 500
              )
          )
          if crop_img
          else transforms.Lambda(lambda img: img)
      ),
      transforms.Resize((image_size, 2 * image_size)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])


class UniversalOCTDataset(datasets.ImageFolder):

  def __init__(
      self, img_dir: str, image_size: int = 224, transform=None, **kwargs
  ):
    super().__init__(root=img_dir, transform=transform)
    print(
        f"Dataset caricato: {len(self)} immagini, classi: {self.class_to_idx}"
    )


UniPD_Dataset = UniversalOCTDataset
UCD_Dataset = UniversalOCTDataset


# ---------------------------------------------------------------- cached


def _load_cached_tensor(path: str) -> torch.Tensor:
  return torch.load(path, map_location="cpu", weights_only=True)


def get_cached_train_transform():
  """Crop and resize are already baked into the cache."""
  return transforms.Compose([
      transforms.ConvertImageDtype(
          torch.float32
      ),  # uint8 -> [0,1], as ToTensor did
      transforms.RandomRotation(15),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ColorJitter(
          brightness=0.1, contrast=0.1, saturation=0.1
      ),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])


def get_cached_val_transform():
  return transforms.Compose([
      transforms.ConvertImageDtype(torch.float32),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
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
    print(
        f"Cached dataset loaded: {len(self)} tensors, classes:"
        f" {self.class_to_idx}"
    )


def read_cache_manifest(cache_path):
  """Returns the manifest dict, or raises if the cache is missing or incomplete."""
  path = Path(cache_path) / "manifest.json"
  if not path.is_file():
    raise FileNotFoundError(
        f"no manifest.json in {cache_path} — run cache_images.py first"
    )
  with open(path) as f:
    return json.load(f)


# ---------------------------------------------------------------- dynamic sampling per round


import re
from pathlib import Path


def _extract_subject_id(filename: str) -> str:
  """Estrae l'ID paziente gestendo:

  - UCD con MRN puro o con Nome:
      '102439_OD Circle (14.2).pt'             -> '102439'
      '102439_JohnDoe_OD Circle (1.0).pt'      -> '102439'
      'JohnDoe_102439_OS Circle (2.0).pt'      -> '102439'
  - UniPD:
      'MS_ID001_306.jpg'                       -> 'ID001'
      'CTRL_ID045_01.jpg'                      -> 'ID045'
  - Formati standard BIDS o generici:
      'sub-042_eye-OD.pt'                      -> 'sub-042'
  """
  stem = Path(filename).stem

  # 1. Se nel file c'è una sequenza numerica tipica da MRN (da 4 a 10 cifre consecutive):
  # Questo cattura l'MRN sia se è all'inizio ("102439_..."), sia se è dopo il nome ("Doe_102439_...")
  mrn_match = re.search(r"(?<!\d)(\d{4,10})(?!\d)", stem)
  if mrn_match:
    return mrn_match.group(1)

  # 2. Pattern alfanumerici espliciti (es. 'ID001', 'sub-042', 'patient12')
  pat_match = re.search(
      r"(ID\d+|SUB[_\-]?\d+|PATIENT\d+|PA\d+)", stem, re.IGNORECASE
  )
  if pat_match:
    return pat_match.group(1)

  # 3. Formato UniPD: scarta la classe iniziale ('MS', 'CTRL', 'HC') se presente
  parts = re.split(r"[_\-]", stem)
  class_labels = {
      "MS",
      "CTRL",
      "HC",
      "CONTROL",
      "CONTROLS",
      "PATIENT",
      "PATIENTS",
  }
  if len(parts) > 1 and parts[0].upper() in class_labels:
    return parts[1]

  # 4. Fallback per nomi con scansione progressiva finale (es. 'paziente_scan1')
  if len(parts) > 1 and parts[-1].isdigit():
    return "_".join(parts[:-1])

  return parts[0] if len(parts) > 1 else stem


def get_subject_stratified_subset(
    dataset, current_round: int, n_ms_subjs: int, n_ctrl_subjs: int
):
  """Campiona esattamente n_ms_subjs e n_ctrl_subjs a rotazione ciclica per round."""
  samples = dataset.samples if hasattr(dataset, "samples") else dataset.imgs

  # Mappa: class_name -> subject_id -> list of indices
  class_to_subjects = defaultdict(lambda: defaultdict(list))
  for idx, (path, class_idx) in enumerate(samples):
    subj_id = _extract_subject_id(path)
    class_name = dataset.classes[class_idx].upper()
    class_to_subjects[class_name][subj_id].append(idx)

  selected_indices = []
  targets_req = {"MS": n_ms_subjs, "CTRL": n_ctrl_subjs}
  selected_counts = {}

  for class_name, req_count in targets_req.items():
    matching_key = next((k for k in class_to_subjects if class_name in k), None)
    if not matching_key:
      continue

    subjects_dict = class_to_subjects[matching_key]
    unique_subjs = sorted(list(subjects_dict.keys()))
    total_subjs = len(unique_subjs)

    actual_take = min(req_count, total_subjs)
    selected_counts[class_name] = actual_take

    # Finestra circolare basata sul round
    start_idx = ((current_round - 1) * actual_take) % total_subjs
    chosen_subjs = [
        unique_subjs[(start_idx + i) % total_subjs] for i in range(actual_take)
    ]

    for subj in chosen_subjs:
      selected_indices.extend(subjects_dict[subj])

  subset = Subset(dataset, selected_indices)
  subset.targets = [dataset.targets[i] for i in selected_indices]

  print(
      f"[Dynamic Subject Sampling] Round {current_round}: selezionati "
      f"{selected_counts.get('MS', 0)} MS e {selected_counts.get('CTRL', 0)} CTRL -> "
      f"{len(selected_indices)} immagini totali."
  )
  return subset