#!/usr/bin/env python3
"""
Run a trained checkpoint over a held-out split and write the metrics.

The checkpoint is either the plain state_dict saved by fl_server.py
(model_best_val_auc.pth / model_federated_final.pth) or a per-site copy saved by
fl_client.py, which records its own backbone and input size and so needs no
--backbone. A bare state_dict does not, and is assumed to be a resnet — which is
what every checkpoint written before that option existed is. Either way nothing
federated happens here: this is a single-process forward pass over one site's
data.

Preprocessing matches the validation path exactly: the same crop rule (UniPD
crops, UCD does not), the same ImageNet normalization, no augmentation. Images
are resized to the model's input size — (image_size, 2*image_size) for the
ResNet, 224x448 for RETFound. Pass --cache-path to read .pt tensors from
cache_images.py instead of decoding PNGs, exactly as the client does; the cache
must contain the requested split.

Scored twice: once per image, and once per patient, pooling each patient's image
scores into a single score (mean by default). The patient id is the leading
'_'-separated token of the filename — '10156650933_OD Circle (419113.002).png'
-> '10156650933' — so both eyes and every repeat scan pool together. Patient
counts are much smaller than image counts, so the patient AUC is the noisier
number, but it is the one that respects the independence of the samples.

Age is read from the cohort's labels.csv (--labels-csv, defaulting to
<data-path>/labels.csv) and keyed by patient, so every scan of a patient carries
that patient's age. Two numbers come out of it, at both image and patient level:
the Spearman rho between age and p(MS), and the AUC of age used on its own as a
classifier. The second is the one that matters — if age alone separates the
classes about as well as the model does, the model may be reading age rather
than disease. Without a usable labels.csv the age analysis is silently skipped
and everything else is unaffected.

Outputs, written to --out-dir:
    predictions.csv    one row per image: path, label, age, p(MS), prediction
    patient_predictions.csv  one row per patient: pooled p(MS), age, n images
    metrics.json       image- and patient-level stats at 0.5 and at Youden,
                       plus the age correlation stats
    metrics.txt        the same image-level numbers in MS_circle's layout
    metrics_patient.txt
    roc.png            ROC over the split
    confusion.png      confusion matrix at 0.5
    confusion_youden.png
    roc_patient.png
    confusion_patient.png
    confusion_patient_youden.png
    scatter_pms_vs_age.png          p(MS) vs age, coloured by true label
    scatter_pms_vs_age_patient.png
    youden/            entropy uncertainty analysis, image level (below)
    youden_patient/    the same, patient level

Entropy uncertainty (entropy_uq.py, the analysis of /data/russ/entropy/youden.py):
the --val-split is scored too, a Youden's J threshold is picked on it — not on
the split being evaluated, unlike the Youden numbers above — and applied to
the evaluated split. Each image / patient gets the binary entropy of its p(MS)
in bits, and the entropy of correct vs incorrect predictions (at the val-picked
threshold) is compared. The patient level re-picks its threshold on the pooled
val scores. Each directory holds youden_threshold.json, val_scan_predictions.csv
(val_patient_...), test_scan_predictions_youden.csv, test_metrics_youden.txt,
entropy_metrics_youden.txt, entropy_dist_correct_vs_incorrect_youden.png and
confusion_matrix_youden.png. Pass --val-split none to skip it.

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
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from model import BACKBONES, RETFOUND_INPUT_HW, build_model
from my_datasets import (CachedOCTDataset, UniversalOCTDataset,
                         get_cached_val_transform, get_val_transform,
                         read_cache_manifest)
from train import (binary_stats, format_stats_block, plot_confusion, plot_roc,
                   youden_threshold)
import entropy_uq

SITES = ("ucd", "unipd")
DEFAULT_CHECKPOINT = "results/090726_test/model_best_val_auc.pth"
DEFAULT_DATA_PATH = "/data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls"


def build_dataset(data_path, split, site, image_size, cache_path, input_hw=None):
    """Returns (dataset, num_workers). Mirrors fl_client.build_datasets' val path.

    image_size is the *cache* size and must match how the cache was built.
    input_hw is the model's input size, applied as a second resize — the two are
    different for RETFound (512-tall cache, 224-tall model).
    """
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
        return CachedOCTDataset(
            split_dir, transform=get_cached_val_transform(input_hw)
        ), 8

    split_dir = os.path.join(data_path, split)
    if not os.path.isdir(split_dir):
        raise SystemExit(f"missing split directory: {split_dir}")
    print(f"[inference] {split}={split_dir} crop={crop} image_size={image_size}")
    return (
        UniversalOCTDataset(
            img_dir=split_dir,
            transform=get_val_transform(image_size, crop_img=crop, input_hw=input_hw),
        ),
        4,
    )


def load_checkpoint(path, device, backbone=None, input_hw=None):
    """Load a state_dict, tolerating the common wrapped layouts.

    Returns (model, spec). The per-site copies written by fl_client.py record
    their own backbone and input size, so they load without being told what
    they are; the server's bare state_dict does not, and falls back to resnet.
    An explicit backbone argument always wins.
    """
    blob = torch.load(path, map_location=device, weights_only=True)

    meta = {}
    if isinstance(blob, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in blob and isinstance(blob[key], dict):
                meta = {k: v for k, v in blob.items() if k != key}
                blob = blob[key]
                break

    if backbone is None:
        backbone = str(meta.get("backbone") or "resnet")
        if backbone not in BACKBONES:
            raise SystemExit(
                f"checkpoint records backbone={backbone!r}, which is not one of "
                f"{list(BACKBONES)}. Pass --backbone explicitly."
            )
    if input_hw is None:
        input_hw = meta.get("input_hw")
    # the resize only applies to retfound; resnet reads the cache size directly
    input_hw = (tuple(input_hw) if input_hw else RETFOUND_INPUT_HW) \
        if backbone == "retfound" else None

    state_dict = OrderedDict(
        (k[len("module."):] if k.startswith("module.") else k, v)
        for k, v in blob.items()
    )
    # Architecture only — the checkpoint supplies every weight.
    model, _ = build_model(
        backbone, weights_path=None, input_hw=input_hw or RETFOUND_INPUT_HW,
        pretrained=False, quiet=True
    )
    model.load_state_dict(state_dict, strict=True)

    spec = {"backbone": backbone, "input_hw": input_hw,
            "round": meta.get("round"), "site": meta.get("site")}
    return model.to(device).eval(), spec


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


def extract_patient_id(path):
    """Leading '_'-separated token of the filename.

    '10156650933_OD Circle (419113.002).png' -> '10156650933'. Every file in the
    UCD cohort matches <digits>_<OD|OS>, so this pools a patient's two eyes and
    any repeat scans together. Names with no '_' fall back to the whole stem,
    which keeps them as singleton patients rather than silently merging them;
    aggregate_by_patient counts those and warns.

    Note this is deliberately not train.py's compute_volume_level_data rule,
    which takes split('_')[1] and would return 'OD Circle (419113.002)' here.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    head, sep, _ = stem.partition("_")
    return (head if sep else stem).strip()


def aggregate_by_patient(samples, y_true, y_score, pooling="mean"):
    """Pool per-image scores into one score per patient.

    samples is the dataset's (path, class_idx) list, in loader order — the
    loader is built with shuffle=False, so it lines up with y_true/y_score.

    Returns (pids, y_true_p, y_score_p, n_images, n_unsplit, conflicts) with the
    patients in first-seen order. conflicts lists any patient carrying more than
    one label, which would mean the same id appeared under both class
    directories; the first label wins and the caller warns.
    """
    pool_fn = {"mean": np.mean, "max": np.max}[pooling]

    order, scores, labels, n_unsplit = [], OrderedDict(), {}, 0
    conflicts = set()
    for (src, _), t, s in zip(samples, y_true, y_score):
        pid = extract_patient_id(src)
        if "_" not in os.path.splitext(os.path.basename(src))[0]:
            n_unsplit += 1
        if pid not in scores:
            scores[pid] = []
            labels[pid] = int(t)
            order.append(pid)
        elif labels[pid] != int(t):
            conflicts.add(pid)
        scores[pid].append(float(s))

    y_score_p = np.array([pool_fn(scores[p]) for p in order], dtype=np.float32)
    y_true_p = np.array([labels[p] for p in order], dtype=np.uint8)
    n_images = np.array([len(scores[p]) for p in order], dtype=np.int32)
    return order, y_true_p, y_score_p, n_images, n_unsplit, sorted(conflicts)


def load_age_map(labels_csv):
    """Read {patient_id: age} from the cohort's labels.csv.

    The CSV carries one row per image with an 'age' column and an
    'arb_person_id' column, which is the same leading '_'-separated token of
    the filename that extract_patient_id returns — so the two key spaces line
    up without any renaming. A patient with several scans at different ages
    (repeat visits) keeps the mean, matching the mean-pooled score.

    Returns {} if the file is missing or has no usable age column, which the
    caller treats as "skip the age analysis" rather than an error: only the UCD
    cohort ships a labels.csv.
    """
    if not labels_csv or not os.path.isfile(labels_csv):
        return {}

    by_pid = OrderedDict()
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "age" not in reader.fieldnames:
            print(f"[inference] {labels_csv} has no 'age' column; "
                  f"skipping the age analysis", file=sys.stderr)
            return {}
        has_pid = "arb_person_id" in reader.fieldnames
        for row in reader:
            pid = (row.get("arb_person_id") or "").strip() if has_pid else ""
            if not pid:
                # fall back to the same rule the predictions use
                pid = extract_patient_id(row.get("filepath") or row.get("path") or "")
            try:
                age = float(row["age"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(age):
                continue
            by_pid.setdefault(pid, []).append(age)

    return {pid: float(np.mean(a)) for pid, a in by_pid.items() if pid}


def ages_for(paths_or_pids, age_map, from_paths):
    """Ages aligned to the given rows, NaN where the patient has none."""
    out = []
    for item in paths_or_pids:
        pid = extract_patient_id(item) if from_paths else item
        out.append(age_map.get(pid, float("nan")))
    return np.asarray(out, dtype=np.float64)


def age_stats(y_true, y_score, ages):
    """Spearman rho between age and p(MS), plus age's own AUC as a baseline.

    Computed on the rows with a known age only; n_with_age says how many that
    was. The age AUC answers "how separable are the classes on age alone" — if
    it is close to the model's AUC, the model may be reading age rather than
    disease.
    """
    m = np.isfinite(ages)
    n = int(m.sum())
    stats = {"n_with_age": n, "n_missing_age": int((~m).sum())}
    if n < 3 or len(np.unique(y_true[m])) < 2:
        stats.update(spearman_rho=float("nan"), spearman_p=float("nan"),
                     age_auc=float("nan"), age_mean=float("nan"))
        return stats

    rho, pval = spearmanr(ages[m], y_score[m])
    try:
        auc = float(roc_auc_score(y_true[m], ages[m]))
    except ValueError:
        auc = float("nan")
    stats.update(spearman_rho=float(rho), spearman_p=float(pval),
                 age_auc=auc, age_mean=float(np.mean(ages[m])))
    return stats


def plot_age_scatter(y_true, y_score, ages, save_path, title, class_names):
    """p(MS) against age, coloured by true label — the confounding check."""
    m = np.isfinite(ages)
    if m.sum() < 3:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue" if t == 0 else "tomato" for t in y_true[m]]
    ax.scatter(ages[m], y_score[m], c=colors, alpha=0.5, s=20, edgecolors="none")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=8, label=class_names[0]),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
               markersize=8, label=class_names[1]),
    ], fontsize=11)
    rho, pval = spearmanr(ages[m], y_score[m])
    ax.set_xlabel("Age at imaging (years)", fontsize=13)
    ax.set_ylabel("p(MS)", fontsize=13)
    ax.set_title(f"{title}  —  Spearman rho={rho:.3f}, p={pval:.3e}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def format_age_block(stats, title):
    return "\n".join([
        f"  {title}",
        f"  {'-' * 54}",
        f"  {'N with known age':<40}{stats['n_with_age']:>12d}",
        f"  {'N missing age':<40}{stats['n_missing_age']:>12d}",
        f"  {'Spearman rho (age vs p(MS))':<40}{stats['spearman_rho']:>12.3f}",
        f"  {'Spearman p-value':<40}{stats['spearman_p']:>12.3e}",
        f"  {'AUC — age alone':<40}{stats['age_auc']:>12.3f}",
    ])


def write_metrics_txt(stats, age, save_path, title, unit="scan",
                      disease_name="MS", extra_lines=None):
    """The MS_circle metrics.txt, same rows in the same order.

    Kept byte-compatible with train_circle.save_metrics_txt so the two projects'
    outputs can be diffed directly: same 40/10 column widths, same labels, same
    3-decimal formatting. The numbers come out of binary_stats and age_stats
    rather than being recomputed here, so this file can never disagree with
    metrics.json. age may be None, in which case the two age rows are dropped.
    """
    lines = [
        title,
        f"{'Metric':<40}  {'Value':>10}",
        f"{'-' * 54}",
        f"{f'N ({unit}s)':<40}  {stats['n']:>10d}",
        f"{'Accuracy':<40}  {stats['accuracy']:>10.3f}",
        f"{f'AUC — {disease_name} prediction ({unit})':<40}  {stats['auc']:>10.3f}",
    ]
    if age is not None:
        lines += [
            f"{'AUC — Age':<40}  {age['age_auc']:>10.3f}",
            f"{f'Spearman ρ (age vs p({disease_name}))':<40}  {age['spearman_rho']:>10.3f}",
            f"{'Spearman p-value':<40}  {age['spearman_p']:>10.3e}",
        ]
    lines += [
        f"{'Balanced Accuracy':<40}  {stats['balanced_accuracy']:>10.3f}",
        f"{'Cohens Kappa':<40}  {stats['kappa']:>10.3f}",
        f"",
        f"{disease_name} class metrics",
        f"  Precision   : {stats['precision_ppv']:.3f}",
        f"  Recall      : {stats['sensitivity_recall']:.3f}",
        f"  Specificity : {stats['specificity']:.3f}",
        f"  F1          : {stats['f1']:.3f}",
        f"  TP={stats['tp']}  FP={stats['fp']}  FN={stats['fn']}  TN={stats['tn']}",
    ]
    if extra_lines:
        lines += [""] + list(extra_lines)
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _age_cell(age):
    return "" if age is None or not np.isfinite(age) else f"{float(age):.4f}"


def write_patient_predictions(path, pids, y_true_p, y_score_p, n_images,
                              threshold, idx_to_class, ages_p):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "label", "class_name", "n_images", "age",
                    "p_ms", "predicted", "predicted_class"])
        for pid, t, s, n, age in zip(pids, y_true_p, y_score_p, n_images, ages_p):
            pred = int(s >= threshold)
            w.writerow([pid, int(t), idx_to_class[int(t)], int(n),
                        _age_cell(age), f"{float(s):.6f}", pred,
                        idx_to_class[pred]])


def write_predictions(path, samples, y_true, y_score, threshold, idx_to_class,
                      ages):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label", "class_name", "age", "p_ms",
                    "predicted", "predicted_class"])
        for (src, _), t, s, age in zip(samples, y_true, y_score, ages):
            pred = int(s >= threshold)
            w.writerow([src, int(t), idx_to_class[int(t)], _age_cell(age),
                        f"{float(s):.6f}", pred, idx_to_class[pred]])


def run_entropy_uq(val, test, out_dir, backbone, pooling):
    """Image- and patient-level youden/entropy reports into out_dir/youden*.

    val and test are dicts of samples, y_true, y_score, ages, pids, y_true_p,
    y_score_p, ages_p — the same arrays main() builds for the evaluated split.
    """
    def frame(ids, id_col, y_true, y_score, ages):
        return pd.DataFrame({id_col: ids, "true_label": y_true.astype(int),
                             "prob_disease": y_score.astype(np.float64),
                             "age": ages})

    summaries = []
    for unit, id_col, sub, ids, yt, ys, ag in (
        ("scan", "filepath", "youden", "paths", "y_true", "y_score", "ages"),
        ("patient", "patient_id", "youden_patient", "pids", "y_true_p",
         "y_score_p", "ages_p"),
    ):
        print(f"\n[inference] entropy UQ — {unit} level")
        try:
            summaries.append(entropy_uq.youden_entropy_report(
                frame(val[ids], id_col, val[yt], val[ys], val[ag]),
                frame(test[ids], id_col, test[yt], test[ys], test[ag]),
                str(out_dir / sub), disease_name="MS", backbone=backbone,
                model_type="FL" if unit == "scan" else f"FL, {pooling}-pooled",
                unit=unit, id_col=id_col))
        except ValueError as e:
            print(f"[inference] WARNING: {unit}-level entropy UQ skipped: {e}",
                  file=sys.stderr)
    if summaries:
        entropy_uq.print_summary_table(summaries, label_key="unit")
    return summaries


def main():
    ap = argparse.ArgumentParser(description="Inference on a held-out OCT split")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help=f"model state_dict (default: {DEFAULT_CHECKPOINT})")
    ap.add_argument("--backbone", default=None, choices=list(BACKBONES),
                    help="architecture the checkpoint was trained with. Read "
                         "from the checkpoint when it records one (the per-site "
                         "copies do), otherwise resnet")
    ap.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                    help="directory holding the split subdirectories")
    ap.add_argument("--split", default="test", help="split to score (default: test)")
    ap.add_argument("--val-split", default="val",
                    help="split the entropy analysis picks its Youden threshold "
                         "on (default: val; 'none' skips the entropy analysis)")
    ap.add_argument("--site", default="ucd", choices=sorted(SITES),
                    help="controls the crop rule: unipd crops, ucd does not")
    ap.add_argument("--cache-path", default=None,
                    help="directory of .pt tensors from cache_images.py; "
                         "must contain the requested split")
    ap.add_argument("--image-size", type=int, default=512,
                    help="cache/decode size; must match how the cache was built")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="overrides the per-source default (4 raw / 8 cached)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="operating point for the confusion matrix and the "
                         "predictions CSV (default: 0.5)")
    ap.add_argument("--patient-pooling", default="mean", choices=("mean", "max"),
                    help="how to pool a patient's image scores into one score "
                         "(default: mean)")
    ap.add_argument("--labels-csv", default=None,
                    help="cohort labels.csv supplying the 'age' column for the "
                         "age vs p(MS) analysis (default: <data-path>/labels.csv "
                         "when it exists; pass 'none' to skip)")
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

    # Load first: the checkpoint may be what tells us the backbone, and the
    # backbone decides the dataset's second resize. RETFound reads 224x448 while
    # the cache stays at its build size, so that is separate from --image-size.
    model, spec = load_checkpoint(ckpt, device, backbone=args.backbone)
    input_hw = spec["input_hw"]
    origin = f" (round {spec['round']} from site {spec['site']})" \
        if spec.get("round") is not None else ""
    print(f"[inference] loaded {ckpt} as {spec['backbone']}{origin}")

    dataset, default_workers = build_dataset(
        args.data_path, args.split, args.site, args.image_size, args.cache_path,
        input_hw=input_hw,
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

    loss, y_true, y_score = predict(model, loader, device)

    thr_youden = youden_threshold(y_true, y_score)
    stats_fixed = binary_stats(y_true, y_score, threshold=args.threshold)
    stats_youden = binary_stats(y_true, y_score, threshold=thr_youden)

    # ---- ages --------------------------------------------------------------
    # Keyed by patient, so every image of a patient carries that patient's age.
    if args.labels_csv and args.labels_csv.lower() == "none":
        labels_csv = None
    elif args.labels_csv:
        labels_csv = args.labels_csv
    else:
        default_csv = os.path.join(args.data_path, "labels.csv")
        labels_csv = default_csv if os.path.isfile(default_csv) else None
    age_map = load_age_map(labels_csv)
    if labels_csv and not age_map:
        print(f"[inference] no usable ages in {labels_csv}; "
              f"the age analysis is skipped", file=sys.stderr)
    elif age_map:
        print(f"[inference] ages for {len(age_map)} patients from {labels_csv}")
    ages = ages_for([s for s, _ in dataset.samples], age_map, from_paths=True)

    class_names = (idx_to_class.get(0, "0"), idx_to_class.get(1, "1"))
    plot_roc(y_true, y_score, out_dir / "roc.png",
             title=f"ROC — {args.site} {args.split}")
    plot_confusion(y_true, y_score, out_dir / "confusion.png",
                   threshold=args.threshold, class_names=class_names)
    plot_confusion(y_true, y_score, out_dir / "confusion_youden.png",
                   threshold=thr_youden, class_names=class_names)
    write_predictions(out_dir / "predictions.csv", dataset.samples, y_true,
                      y_score, args.threshold, idx_to_class, ages)

    age_stats_img = age_stats(y_true, y_score, ages) if age_map else None
    if age_map:
        plot_age_scatter(y_true, y_score, ages,
                         out_dir / "scatter_pms_vs_age.png",
                         f"p(MS) vs Age — {args.site} {args.split} (image)",
                         class_names)

    # ---- patient level -----------------------------------------------------
    pids, y_true_p, y_score_p, n_images, n_unsplit, conflicts = \
        aggregate_by_patient(dataset.samples, y_true, y_score,
                             pooling=args.patient_pooling)
    if n_unsplit:
        print(f"[inference] WARNING: {n_unsplit} filename(s) contain no '_'; "
              f"each was kept as its own patient, so the patient-level numbers "
              f"may be closer to the image-level ones than they look",
              file=sys.stderr)
    if conflicts:
        print(f"[inference] WARNING: {len(conflicts)} patient id(s) appear under "
              f"both class directories, e.g. {conflicts[:5]}; the first label "
              f"seen wins", file=sys.stderr)

    # Youden is re-derived on the pooled scores: pooling shifts the score
    # distribution, so the image-level threshold does not carry over.
    thr_youden_p = youden_threshold(y_true_p, y_score_p)
    stats_fixed_p = binary_stats(y_true_p, y_score_p, threshold=args.threshold)
    stats_youden_p = binary_stats(y_true_p, y_score_p, threshold=thr_youden_p)

    plot_roc(y_true_p, y_score_p, out_dir / "roc_patient.png",
             title=f"ROC (patient) — {args.site} {args.split}")
    plot_confusion(y_true_p, y_score_p, out_dir / "confusion_patient.png",
                   threshold=args.threshold, class_names=class_names)
    plot_confusion(y_true_p, y_score_p, out_dir / "confusion_patient_youden.png",
                   threshold=thr_youden_p, class_names=class_names)
    ages_p = ages_for(pids, age_map, from_paths=False)
    write_patient_predictions(out_dir / "patient_predictions.csv", pids,
                              y_true_p, y_score_p, n_images, args.threshold,
                              idx_to_class, ages_p)

    age_stats_pat = age_stats(y_true_p, y_score_p, ages_p) if age_map else None
    if age_map:
        plot_age_scatter(y_true_p, y_score_p, ages_p,
                         out_dir / "scatter_pms_vs_age_patient.png",
                         f"p(MS) vs Age — {args.site} {args.split} (patient)",
                         class_names)

    # ---- metrics.txt -------------------------------------------------------
    # The Youden operating point is carried as extra lines rather than a second
    # file: the fixed threshold is the headline, but Youden is what the ROC-based
    # comparisons in the other project quote.
    bb, pool = spec["backbone"], args.patient_pooling
    write_metrics_txt(
        stats_fixed, age_stats_img, out_dir / "metrics.txt",
        title=f"MS Classifier — {args.site} {args.split} IMAGE-LEVEL "
              f"({bb}, threshold {args.threshold:g})",
        unit="image",
        extra_lines=[
            "At the Youden threshold",
            f"  Threshold            : {thr_youden:.3f}",
            f"  Balanced Accuracy    : {stats_youden['balanced_accuracy']:.3f}",
            f"  Sensitivity          : {stats_youden['sensitivity_recall']:.3f}",
            f"  Specificity          : {stats_youden['specificity']:.3f}",
            "",
            f"Cross-entropy loss     : {loss:.4f}",
            f"Checkpoint             : {ckpt.resolve()}",
        ],
    )
    write_metrics_txt(
        stats_fixed_p, age_stats_pat, out_dir / "metrics_patient.txt",
        title=f"MS Classifier — {args.site} {args.split} PATIENT-LEVEL "
              f"({pool} over scans, {bb}, threshold {args.threshold:g})",
        unit="patient",
        extra_lines=[
            "Patient-level aggregation",
            f"  Aggregator used      : {pool}",
            f"  Images               : {len(dataset)}",
            f"  Patients             : {len(pids)}",
            f"  Patients with >1 image: {int((n_images > 1).sum())}  "
            f"(max {int(n_images.max())}, mean {n_images.mean():.2f})",
            "",
            "At the Youden threshold",
            f"  Threshold            : {thr_youden_p:.3f}",
            f"  Balanced Accuracy    : {stats_youden_p['balanced_accuracy']:.3f}",
            f"  Sensitivity          : {stats_youden_p['sensitivity_recall']:.3f}",
            f"  Specificity          : {stats_youden_p['specificity']:.3f}",
            "",
            f"Checkpoint             : {ckpt.resolve()}",
        ],
    )

    payload = {
        "checkpoint": str(ckpt.resolve()),
        "backbone": spec["backbone"],
        "model_input_hw": list(input_hw) if input_hw else None,
        "checkpoint_round": spec["round"],
        "checkpoint_site": spec["site"],
        "data_path": args.data_path,
        "cache_path": args.cache_path,
        "site": args.site,
        "split": args.split,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "class_to_idx": dataset.class_to_idx,
        "n_images": len(dataset),
        "n_patients": len(pids),
        "patient_pooling": args.patient_pooling,
        "loss": float(loss),
        "youden_threshold": float(thr_youden),
        f"stats_at_{args.threshold:g}": stats_fixed,
        "stats_at_youden": stats_youden,
        "patient_youden_threshold": float(thr_youden_p),
        f"patient_stats_at_{args.threshold:g}": stats_fixed_p,
        "patient_stats_at_youden": stats_youden_p,
        # Flat copies of the headline numbers. Everything here is already in the
        # stats blocks above; this is purely so a sweep over many runs can read
        # them without knowing the threshold-dependent key names.
        "summary": {
            "image_auc": stats_fixed["auc"],
            "patient_auc": stats_fixed_p["auc"],
            "image_balanced_accuracy": stats_fixed["balanced_accuracy"],
            "image_balanced_accuracy_youden": stats_youden["balanced_accuracy"],
            "patient_balanced_accuracy": stats_fixed_p["balanced_accuracy"],
            "patient_balanced_accuracy_youden": stats_youden_p["balanced_accuracy"],
        },
        "labels_csv": labels_csv,
        "age_stats": age_stats_img,
        "patient_age_stats": age_stats_pat,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    tag = f"{args.site} {args.split}"
    print()
    print(format_stats_block(stats_fixed,
                             f"{tag} — image — threshold {args.threshold:g}"))
    print()
    print(format_stats_block(stats_youden, f"{tag} — image — Youden threshold"))
    print()
    print(format_stats_block(
        stats_fixed_p,
        f"{tag} — patient ({pool}-pooled) — threshold {args.threshold:g}"))
    print()
    print(format_stats_block(
        stats_youden_p, f"{tag} — patient ({pool}-pooled) — Youden threshold"))
    if age_stats_img is not None:
        print()
        print(format_age_block(age_stats_img, f"{tag} — image — age vs p(MS)"))
        print()
        print(format_age_block(
            age_stats_pat, f"{tag} — patient ({pool}-pooled) — age vs p(MS)"))
    print()
    print(f"  {len(dataset)} images pooled into {len(pids)} patients "
          f"({n_images.min()}-{n_images.max()} images each, "
          f"median {int(np.median(n_images))})")
    print(f"  Image AUC{'':<23}{stats_fixed['auc']:>12.4f}")
    print(f"  Patient AUC ({pool}-pooled){'':<8}{stats_fixed_p['auc']:>12.4f}")
    # Balanced accuracy at both thresholds: at 0.5 it is the number the confusion
    # matrices show, but on a cohort this unbalanced (145 MS / 675 controls) the
    # Youden one is the fairer read of what the model can do.
    print(f"  Image bAcc @{args.threshold:<g}{'':<18}"
          f"{stats_fixed['balanced_accuracy']:>12.4f}")
    print(f"  Image bAcc @Youden{'':<14}"
          f"{stats_youden['balanced_accuracy']:>12.4f}")
    print(f"  Patient bAcc @{args.threshold:<g}{'':<16}"
          f"{stats_fixed_p['balanced_accuracy']:>12.4f}")
    print(f"  Patient bAcc @Youden{'':<12}"
          f"{stats_youden_p['balanced_accuracy']:>12.4f}")
    print(f"  Cross-entropy loss{'':<14}{loss:>12.4f}")
    if age_stats_pat is not None:
        print(f"  Age AUC (patient){'':<15}{age_stats_pat['age_auc']:>12.4f}")
        print(f"  Spearman rho age~p(MS){'':<10}"
              f"{age_stats_pat['spearman_rho']:>12.4f}")

    # ---- entropy uncertainty, Youden threshold picked on --val-split -------
    if args.val_split.lower() != "none":
        if args.val_split == args.split:
            raise SystemExit("--val-split must differ from --split: the threshold "
                             "has to be picked on held-out data")
        val_ds, _ = build_dataset(args.data_path, args.val_split, args.site,
                                  args.image_size, args.cache_path,
                                  input_hw=input_hw)
        if val_ds.class_to_idx != dataset.class_to_idx:
            raise SystemExit(f"class_to_idx differs between {args.val_split} "
                             f"({val_ds.class_to_idx}) and {args.split} "
                             f"({dataset.class_to_idx})")
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=workers,
                                pin_memory=device.type == "cuda")
        _, vy_true, vy_score = predict(model, val_loader, device)
        v_paths = [s for s, _ in val_ds.samples]
        v_pids, vy_true_p, vy_score_p, _, _, _ = aggregate_by_patient(
            val_ds.samples, vy_true, vy_score, pooling=args.patient_pooling)
        val = dict(paths=v_paths, y_true=vy_true, y_score=vy_score,
                   ages=ages_for(v_paths, age_map, from_paths=True),
                   pids=v_pids, y_true_p=vy_true_p, y_score_p=vy_score_p,
                   ages_p=ages_for(v_pids, age_map, from_paths=False))
        test = dict(paths=[s for s, _ in dataset.samples], y_true=y_true,
                    y_score=y_score, ages=ages, pids=pids, y_true_p=y_true_p,
                    y_score_p=y_score_p, ages_p=ages_p)
        run_entropy_uq(val, test, out_dir, bb, pool)

    print(f"\n[inference] wrote {out_dir}")


if __name__ == "__main__":
    main()
