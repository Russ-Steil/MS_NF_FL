"""
Training and evaluation functions for the ResNet50 binary classifier.
Used by the Flower client for local fit and evaluate.
"""
import os
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score, roc_curve, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def train(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 0.0005,
    weight_decay: float = 0.005,
    class_weights: Optional[torch.Tensor] = None
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    for epoch in range(epochs):
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", total=len(trainloader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate(
    model: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Returns (loss, accuracy, y_true, y_score).
    y_score is P(class 1) = P(MS), needed by the server to pool a global ROC.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    correct, total, loss_sum, n_batches = 0, 0, 0.0, 0
    all_scores, all_labels = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item()
            n_batches += 1

            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_scores.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total else 0.0
    loss = loss_sum / n_batches if n_batches else 0.0
    y_score = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    return float(loss), accuracy, y_true.astype(np.uint8), y_score.astype(np.float32)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """roc_auc_score raises if only one class is present. Return NaN instead."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def binary_stats(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Full binary classification stats at a given decision threshold."""
    y_true = y_true.astype(int)
    preds = (y_score >= threshold).astype(int)

    tp = int(((y_true == 1) & (preds == 1)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    fn = int(((y_true == 1) & (preds == 0)).sum())
    tn = int(((y_true == 0) & (preds == 0)).sum())

    precision   = tp / (tp + fp) if (tp + fp) else 0.0
    recall      = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv         = tn / (tn + fn) if (tn + fn) else 0.0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc         = (tp + tn) / len(y_true) if len(y_true) else 0.0

    try:
        bacc = float(balanced_accuracy_score(y_true, preds))
    except ValueError:
        bacc = float('nan')
    try:
        kappa = float(cohen_kappa_score(y_true, preds))
    except ValueError:
        kappa = float('nan')

    return {
        "threshold": float(threshold),
        "auc": safe_auc(y_true, y_score),
        "accuracy": float(acc),
        "balanced_accuracy": bacc,
        "kappa": kappa,
        "sensitivity_recall": float(recall),
        "specificity": float(specificity),
        "precision_ppv": float(precision),
        "npv": float(npv),
        "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
    }


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold maximizing sensitivity + specificity - 1."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return float(thr[int(np.argmax(tpr - fpr))])


def format_stats_block(stats: Dict[str, float], title: str) -> str:
    lines = [
        title,
        f"{'-' * 54}",
        f"{'Threshold':<32}{stats['threshold']:>12.3f}",
        f"{'AUC':<32}{stats['auc']:>12.4f}",
        f"{'Accuracy':<32}{stats['accuracy']:>12.4f}",
        f"{'Balanced accuracy':<32}{stats['balanced_accuracy']:>12.4f}",
        f"{'Cohen kappa':<32}{stats['kappa']:>12.4f}",
        f"{'Sensitivity (recall)':<32}{stats['sensitivity_recall']:>12.4f}",
        f"{'Specificity':<32}{stats['specificity']:>12.4f}",
        f"{'Precision (PPV)':<32}{stats['precision_ppv']:>12.4f}",
        f"{'NPV':<32}{stats['npv']:>12.4f}",
        f"{'F1':<32}{stats['f1']:>12.4f}",
        "",
        f"  TP={stats['tp']}  FP={stats['fp']}  FN={stats['fn']}  TN={stats['tn']}",
        f"  n={stats['n']}  positives={stats['n_pos']}  negatives={stats['n_neg']}",
    ]
    return "\n".join(lines)


def plot_loss_curves(train_loss, val_loss, save_path):
    rounds = range(1, len(val_loss) + 1)
    plt.figure(figsize=(7, 5))
    if len(train_loss) == len(val_loss) and train_loss:
        plt.plot(rounds, train_loss, '-o', color='steelblue', label='Train loss')
    plt.plot(rounds, val_loss, '-o', color='tomato', label='Val loss')
    plt.xlabel('Federated round')
    plt.ylabel('Cross-entropy loss')
    plt.title('Loss curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_auc_curves(train_auc, val_auc, save_path):
    rounds = range(1, len(val_auc) + 1)
    plt.figure(figsize=(7, 5))
    if len(train_auc) == len(val_auc) and train_auc:
        plt.plot(rounds, train_auc, '-o', color='steelblue', label='Train AUC')
    plt.plot(rounds, val_auc, '-o', color='tomato', label='Val AUC')
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
    plt.ylim(0.3, 1.02)
    plt.xlabel('Federated round')
    plt.ylabel('AUC')
    plt.title('AUC progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc(y_true, y_score, save_path, title='ROC — validation (pooled across sites)'):
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        print("  [roc] skipped — only one class present")
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='tomato', linewidth=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Chance')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.xlabel('1 - specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion(y_true, y_score, save_path, threshold=0.5, class_names=('CTRL', 'MS')):
    preds = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true.astype(int), preds, labels=[0, 1])
    plt.figure(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(class_names), yticklabels=list(class_names),
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title(f'Confusion matrix (threshold={threshold:.2f})', fontsize=13, pad=12)
    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('True', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# Volume-level / age-correlation utilities. Unchanged from the original file.
# Currently unused — evaluate_and_plot_volume is never called by the client.
# ============================================================================

def load_age_dict(excel_path: str) -> Dict[str, float]:
    try:
        df = pd.read_excel(excel_path)
        df['ID_Paziente'] = df['ID_Paziente'].astype(str).str.strip()
        age_dict = dict(zip(df['ID_Paziente'], df['Eta'].astype(float)))
        print(f"Excel file loaded with {len(age_dict)} patients.")
        return age_dict
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return {}


def compute_volume_level_data(loader, dataset, model, age_dict, device):
    model.eval()
    patient_probs, patient_labels, patient_ages = {}, {}, {}
    file_paths = [sample[0] for sample in dataset.samples]
    file_idx = 0

    print("Processing test data for volume-level evaluation...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            labels = labels.numpy()

            for i in range(len(labels)):
                full_path = file_paths[file_idx]
                file_name = os.path.basename(full_path)
                try:
                    patient_id = file_name.split('_')[1].strip()
                except IndexError:
                    patient_id = "UNKNOWN"
                age = age_dict.get(patient_id, float('nan'))
                prob_ms = probs[i][1]
                if patient_id not in patient_probs:
                    patient_probs[patient_id] = []
                    patient_labels[patient_id] = labels[i]
                    patient_ages[patient_id] = age
                patient_probs[patient_id].append(prob_ms)
                file_idx += 1

    final_labels, final_probs_ms, final_ages = [], [], []
    for p_id in patient_probs.keys():
        if np.isnan(patient_ages[p_id]):
            continue
        final_labels.append(patient_labels[p_id])
        final_probs_ms.append(np.mean(patient_probs[p_id]))
        final_ages.append(patient_ages[p_id])
    return np.array(final_labels), np.array(final_probs_ms), np.array(final_ages)


def save_scatter_plot(probs_ms, ages, labels, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue' if l == 0 else 'tomato' for l in labels]
    ax.scatter(ages, probs_ms, c=colors, alpha=0.5, s=20, edgecolors='none')
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='Control'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', markersize=8, label='MS'),
    ], fontsize=11)
    rho, pval = spearmanr(ages, probs_ms)
    ax.set_xlabel('Age at imaging (years)', fontsize=13)
    ax.set_ylabel('p(MS)', fontsize=13)
    ax.set_title(f'p(MS) vs Age  —  Spearman rho={rho:.3f}, p={pval:.3e}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Scatter plot saved to {save_path}")