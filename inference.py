#!/usr/bin/env python3
"""
Run a trained checkpoint over a held-out split and write the metrics.

The checkpoint is a plain ResNet50 state_dict as saved by fl_server.py
(model_best_val_auc.pth / model_federated_final.pth), so nothing federated is
involved here — this is a single-process forward pass over one site's data.

Preprocessing matches the validation path exactly: the same crop rule (UniPD
crops, UCD does not), the same resize to (image_size, 2*image_size), the same
ImageNet normalization, no augmentation. Pass --cache-path to read .pt tensors
from cache_images.py instead of decoding PNGs, exactly as the client does; the
cache must contain the requested split.

Outputs, written to --out-dir:
    predictions.csv    one row per image: path, label, p(MS), prediction
    metrics.json       image-level stats at 0.5 and at the Youden threshold
    roc.png            ROC over the split
    confusion.png      confusion matrix at 0.5
    confusion_youden.png

Example:
    python inference.py \
      --checkpoint results/090726_test/model_best_val_auc.pth \
      --data-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls \
      --site ucd --split test
"""
import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_resnet50_binary
from my_datasets import (CachedOCTDataset, UniversalOCTDataset,
                         get_cached_val_transform, get_val_transform,
                         read_cache_manifest)
from train import (binary_stats, format_stats_block, plot_confusion, plot_roc,
                   youden_threshold)

SITES = ("ucd", "unipd")
DEFAULT_CHECKPOINT = "results/090726_test/model_best_val_auc.pth"
DEFAULT_DATA_PATH = "/data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls"


def build_dataset(data_path, split, site, image_size, cache_path):
    """Returns (dataset, num_workers). Mirrors fl_client.build_datasets' val path."""
    crop = site.lower() == "unipd"

    if cache_path:
        manifest = read_cache_manifest(cache_path)
        if bool(manifest["crop"]) != crop:
            raise SystemExit(
                f"cache at {cache_path} was built with crop={manifest['crop']} "
                f"but site '{site}' needs crop={crop}"
            )
        if int(manifest["image_size"]) != int(image_size):
            raise SystemExit(
                f"cache at {cache_path} was built at image_size="
                f"{manifest['image_size']} but this run wants {image_size}"
            )
        split_dir = os.path.join(cache_path, split)
        if not os.path.isdir(split_dir):
            raise SystemExit(
                f"no '{split}' split in the cache at {cache_path} — "
                f"cache_images.py only writes train/ and val/, so either cache "
                f"the split first or drop --cache-path to decode PNGs"
            )
        print(f"[inference] cached {split}={split_dir} (built {manifest['created_at']})")
        return CachedOCTDataset(split_dir, transform=get_cached_val_transform()), 8

    split_dir = os.path.join(data_path, split)
    if not os.path.isdir(split_dir):
        raise SystemExit(f"missing split directory: {split_dir}")
    print(f"[inference] {split}={split_dir} crop={crop} image_size={image_size}")
    return (
        UniversalOCTDataset(
            img_dir=split_dir, transform=get_val_transform(image_size, crop_img=crop)
        ),
        4,
    )


def load_checkpoint(path, device):
    """Load a bare state_dict, tolerating the common wrapped layouts."""
    blob = torch.load(path, map_location=device, weights_only=True)
    if isinstance(blob, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in blob and isinstance(blob[key], dict):
                blob = blob[key]
                break
    state_dict = OrderedDict(
        (k[len("module."):] if k.startswith("module.") else k, v)
        for k, v in blob.items()
    )
    model = get_resnet50_binary(pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, loader, device):
    """Returns (loss, y_true, y_score) with y_score = P(class 1) = P(MS)."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, n = 0.0, 0
    scores, labels = [], []

    for images, targets in tqdm(loader, desc="inference", total=len(loader)):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, targets).item()
        n += targets.size(0)
        scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(targets.cpu().numpy())

    y_score = np.concatenate(scores).astype(np.float32) if scores else np.array([], np.float32)
    y_true = np.concatenate(labels).astype(np.uint8) if labels else np.array([], np.uint8)
    return (loss_sum / n if n else float("nan")), y_true, y_score


def write_predictions(path, samples, y_true, y_score, threshold, idx_to_class):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label", "class_name", "p_ms", "predicted",
                    "predicted_class"])
        for (src, _), t, s in zip(samples, y_true, y_score):
            pred = int(s >= threshold)
            w.writerow([src, int(t), idx_to_class[int(t)], f"{float(s):.6f}",
                        pred, idx_to_class[pred]])


def main():
    ap = argparse.ArgumentParser(description="Inference on a held-out OCT split")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help=f"ResNet50 state_dict (default: {DEFAULT_CHECKPOINT})")
    ap.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                    help="directory holding the split subdirectories")
    ap.add_argument("--split", default="test", help="split to score (default: test)")
    ap.add_argument("--site", default="ucd", choices=sorted(SITES),
                    help="controls the crop rule: unipd crops, ucd does not")
    ap.add_argument("--cache-path", default=None,
                    help="directory of .pt tensors from cache_images.py; "
                         "must contain the requested split")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="overrides the per-source default (4 raw / 8 cached)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="operating point for the confusion matrix and the "
                         "predictions CSV (default: 0.5)")
    ap.add_argument("--out-dir", default=None,
                    help="default: <checkpoint dir>/inference_<split>")
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"no checkpoint at {ckpt}")

    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent / f"inference_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        if args.gpu_index >= torch.cuda.device_count():
            raise SystemExit(
                f"gpu_index={args.gpu_index} but only "
                f"{torch.cuda.device_count()} CUDA device(s) visible"
            )
        device = torch.device(f"cuda:{args.gpu_index}")
        print(f"[inference] using {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device("cpu")
        print("[inference] no CUDA available, using cpu")

    dataset, default_workers = build_dataset(
        args.data_path, args.split, args.site, args.image_size, args.cache_path
    )
    if len(dataset) == 0:
        raise SystemExit(f"the '{args.split}' split is empty")

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    if dataset.class_to_idx.get("MS") != 1:
        print(f"[inference] WARNING: class_to_idx is {dataset.class_to_idx}; "
              f"the model was trained with MS as class 1, so scores would be "
              f"inverted", file=sys.stderr)

    workers = default_workers if args.num_workers is None else args.num_workers
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")

    model = load_checkpoint(ckpt, device)
    print(f"[inference] loaded {ckpt}")

    loss, y_true, y_score = predict(model, loader, device)

    thr_youden = youden_threshold(y_true, y_score)
    stats_fixed = binary_stats(y_true, y_score, threshold=args.threshold)
    stats_youden = binary_stats(y_true, y_score, threshold=thr_youden)

    class_names = (idx_to_class.get(0, "0"), idx_to_class.get(1, "1"))
    plot_roc(y_true, y_score, out_dir / "roc.png",
             title=f"ROC — {args.site} {args.split}")
    plot_confusion(y_true, y_score, out_dir / "confusion.png",
                   threshold=args.threshold, class_names=class_names)
    plot_confusion(y_true, y_score, out_dir / "confusion_youden.png",
                   threshold=thr_youden, class_names=class_names)
    write_predictions(out_dir / "predictions.csv", dataset.samples, y_true,
                      y_score, args.threshold, idx_to_class)

    payload = {
        "checkpoint": str(ckpt.resolve()),
        "data_path": args.data_path,
        "cache_path": args.cache_path,
        "site": args.site,
        "split": args.split,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "class_to_idx": dataset.class_to_idx,
        "n_images": len(dataset),
        "loss": float(loss),
        "youden_threshold": float(thr_youden),
        f"stats_at_{args.threshold:g}": stats_fixed,
        "stats_at_youden": stats_youden,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    print()
    print(format_stats_block(stats_fixed,
                             f"{args.site} {args.split} — threshold {args.threshold:g}"))
    print()
    print(format_stats_block(stats_youden,
                             f"{args.site} {args.split} — Youden threshold"))
    print()
    print(f"  Cross-entropy loss{'':<14}{loss:>12.4f}")
    print(f"\n[inference] wrote {out_dir}")


if __name__ == "__main__":
    main()
