#!/usr/bin/env python3
"""
Pre-cache OCT images as uint8 tensors to remove PIL decode + resize from the
training loop. Run once per site before training.

What is cached: the deterministic prefix of the transform pipeline, i.e. the
UniPD left-crop (when applicable) and the resize, stored as a uint8 CHW tensor.
This is bit-identical to ToTensor's output once divided by 255 at load time.
Rotation, flip, jitter and normalization stay in the dataloader because they
must vary per epoch.

The output mirrors the source tree, so DatasetFolder still discovers classes
and targets the same way ImageFolder did:

    <cache-path>/train/<class>/<name>.pt
    <cache-path>/val/<class>/<name>.pt
    <cache-path>/manifest.json

manifest.json records the crop flag and image size. fl_client.py refuses to
train against a cache whose manifest disagrees with the run config, so a stale
cache cannot silently change your inputs.
"""
import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
SPLITS = ("train", "val")
CROP_LEFT = 400


def build_task_list(src_root, dst_root, splits):
    """Mirror <src>/<split>/<class>/<image> to <dst>/<split>/<class>/<stem>.pt."""
    tasks = []
    for split in splits:
        split_dir = src_root / split
        if not split_dir.is_dir():
            raise SystemExit(f"missing split directory: {split_dir}")

        classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        if not classes:
            raise SystemExit(f"no class subdirectories under {split_dir}")

        for cls in classes:
            out_dir = dst_root / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for entry in sorted((split_dir / cls).iterdir()):
                if entry.is_file() and entry.suffix.lower() in IMG_EXTENSIONS:
                    tasks.append((str(entry), str(out_dir / f"{entry.stem}.pt")))
    return tasks


def encode_one(args):
    """
    Returns (status, src, detail).
    status is one of: ok, skipped, failed.
    """
    src, dst, image_size, crop, force = args
    try:
        if not force and os.path.exists(dst):
            return ("skipped", src, "")

        with Image.open(src) as im:
            im = im.convert("RGB")

            if crop:
                if im.width <= CROP_LEFT:
                    return ("failed", src,
                            f"width {im.width} <= crop of {CROP_LEFT}px")
                im = im.crop((CROP_LEFT, 0, im.width, im.height))

            # PIL resize, matching transforms.Resize((h, 2h)) on a PIL input
            im = im.resize((2 * image_size, image_size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)

        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()

        tmp = dst + ".tmp"
        torch.save(tensor, tmp)
        os.replace(tmp, dst)
        return ("ok", src, "")

    except Exception as exc:
        return ("failed", src, f"{type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser(
        description="Cache OCT images as uint8 tensors for federated training"
    )
    ap.add_argument("--site", required=True, choices=["ucd", "unipd"],
                    help="determines whether the 400px left crop is applied")
    ap.add_argument("--data-path", required=True,
                    help="source directory holding train/ and val/")
    ap.add_argument("--cache-path", required=True,
                    help="destination directory for the .pt cache")
    ap.add_argument("--image-size", type=int, default=512,
                    help="output is image_size x 2*image_size (default 512)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--force", action="store_true",
                    help="re-encode images that are already cached")
    args = ap.parse_args()

    src_root = Path(args.data_path)
    dst_root = Path(args.cache_path)
    crop = args.site == "unipd"

    print(f"[cache:{args.site}] source     {src_root}")
    print(f"[cache:{args.site}] cache      {dst_root}")
    print(f"[cache:{args.site}] image_size {args.image_size} "
          f"(output {args.image_size}x{2 * args.image_size})")
    print(f"[cache:{args.site}] crop       {crop}"
          + (f" (left {CROP_LEFT}px)" if crop else ""))
    print(f"[cache:{args.site}] workers    {args.workers}")
    print(f"[cache:{args.site}] force      {args.force}", flush=True)

    dst_root.mkdir(parents=True, exist_ok=True)
    tasks = build_task_list(src_root, dst_root, SPLITS)
    if not tasks:
        raise SystemExit("found no images to cache")

    per_image_mb = (3 * args.image_size * 2 * args.image_size) / 1024**2
    print(f"[cache:{args.site}] {len(tasks)} images, "
          f"~{per_image_mb:.2f} MiB each, "
          f"~{len(tasks) * per_image_mb / 1024:.1f} GiB total", flush=True)

    payloads = [(s, d, args.image_size, crop, args.force) for s, d in tasks]

    t0 = time.time()
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(encode_one, p) for p in payloads]
        for i, fut in enumerate(as_completed(futures), 1):
            status, src, detail = fut.result()
            counts[status] += 1
            if status == "failed":
                failures.append((src, detail))
            if i % 500 == 0 or i == len(futures):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"[cache:{args.site}] {i}/{len(futures)}  "
                      f"ok={counts['ok']} skipped={counts['skipped']} "
                      f"failed={counts['failed']}  {rate:.0f} img/s", flush=True)

    elapsed = time.time() - t0

    # per-split counts for the manifest
    split_counts = {}
    for split in SPLITS:
        n = sum(1 for _ in (dst_root / split).rglob("*.pt"))
        split_counts[split] = n

    manifest = {
        "site": args.site,
        "crop": crop,
        "crop_left": CROP_LEFT if crop else 0,
        "image_size": args.image_size,
        "output_shape": [3, args.image_size, 2 * args.image_size],
        "dtype": "uint8",
        "source_path": str(src_root.resolve()),
        "cache_path": str(dst_root.resolve()),
        "counts": split_counts,
        "encoded": counts["ok"],
        "skipped": counts["skipped"],
        "failed": counts["failed"],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "torch": torch.__version__,
    }
    with open(dst_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[cache:{args.site}] done in {elapsed/60:.1f} min — "
          f"encoded={counts['ok']} skipped={counts['skipped']} "
          f"failed={counts['failed']}")
    for split, n in split_counts.items():
        print(f"[cache:{args.site}]   {split}: {n} cached tensors")
    print(f"[cache:{args.site}] manifest -> {dst_root / 'manifest.json'}")

    if failures:
        print(f"[cache:{args.site}] {len(failures)} failure(s):", file=sys.stderr)
        for src, detail in failures[:20]:
            print(f"    {src}: {detail}", file=sys.stderr)
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()