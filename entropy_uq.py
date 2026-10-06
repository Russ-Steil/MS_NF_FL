"""
Youden's-J threshold selection + entropy uncertainty analysis, shared by
MS_NF_FL/inference.py and MS_circle/inference_retfound_224.py.

The analysis is /data/russ/entropy/youden.py's, step for step. The helpers below
are copied from it verbatim, with two exceptions: the scoring unit ("volume" in
youden.py) is a parameter, so the same code serves scan- and patient-level
scores, and the run orchestration is replaced by youden_entropy_report, which
takes predictions that have already been made. Keep the two copies of this file
(MS_NF_FL/ and MS_circle/) identical. MS_NF_FL ships to the sites on its own, so
it cannot import from /data/russ/entropy.

  1. Pick the Youden's J threshold on val (argmax TPR - FPR)
  2. Apply it to test; recompute metrics and correct/incorrect
  3. Binary predictive entropy in bits per test unit, correct vs incorrect

Outputs, written to out_dir (unit = "volume" reproduces youden.py's names):
  youden_threshold.json
  val_{unit}_predictions.csv
  test_{unit}_predictions_youden.csv
  test_metrics_youden.txt                        metrics at 0.5 vs Youden threshold
  entropy_metrics_youden.txt                     entropy, correct vs incorrect (Youden)
  entropy_dist_correct_vs_incorrect_youden.png
  confusion_matrix_youden.png
"""

import json
import os

import numpy as np
from sklearn.metrics import (roc_auc_score, roc_curve, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# Threshold / metric helpers (from youden.py)
# ══════════════════════════════════════════════════════════════════════════════

def entropy_bits(p_disease: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p   = np.stack([1.0 - p_disease, p_disease], axis=1)
    p   = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=1)


def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> dict:
    fpr, tpr, thr = roc_curve(labels, probs)
    finite        = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    j = tpr - fpr
    i = int(np.argmax(j))
    return {
        'threshold':       float(thr[i]),
        'youden_j':        float(j[i]),
        'val_sensitivity': float(tpr[i]),
        'val_specificity': float(1.0 - fpr[i]),
        'val_auc':         float(roc_auc_score(labels, probs)),
        'n_val':           int(len(labels)),
        'n_val_pos':       int((labels == 1).sum()),
    }


def threshold_metrics(labels: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    labels = labels.astype(int)
    preds  = (probs >= thr).astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
    return {
        'threshold':   float(thr),
        'accuracy':    float((preds == labels).mean()),
        'bacc':        float(balanced_accuracy_score(labels, preds)),
        'kappa':       float(cohen_kappa_score(labels, preds)),
        'sensitivity': float(recall),
        'specificity': float(specificity),
        'precision':   float(precision),
        'f1':          float(f1),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def save_metrics_txt(test_auc, m05, my, yinfo, save_path, title):
    rows = [('Threshold', 'threshold', '.4f'), ('Accuracy', 'accuracy', '.3f'),
            ('Balanced Accuracy', 'bacc', '.3f'), ('Cohens Kappa', 'kappa', '.3f'),
            ('Sensitivity', 'sensitivity', '.3f'), ('Specificity', 'specificity', '.3f'),
            ('Precision', 'precision', '.3f'), ('F1', 'f1', '.3f'),
            ('TP', 'tp', 'd'), ('FP', 'fp', 'd'), ('FN', 'fn', 'd'), ('TN', 'tn', 'd')]
    lines = [
        title,
        f"Youden threshold picked on val: {yinfo['threshold']:.4f}  "
        f"(J={yinfo['youden_j']:.3f}, val sens={yinfo['val_sensitivity']:.3f}, "
        f"val spec={yinfo['val_specificity']:.3f}, val AUC={yinfo['val_auc']:.3f})",
        f"",
        f"{'Test AUC (threshold-free)':<30} {test_auc:>10.3f}",
        f"{'-'*54}",
        f"{'Metric':<30} {'@0.5':>10} {'@Youden':>12}",
        f"{'-'*54}",
    ]
    for name, key, fmt in rows:
        lines.append(f"{name:<30} {format(m05[key], fmt):>10} {format(my[key], fmt):>12}")
    txt = "\n".join(lines)
    print(f"\n{txt}")
    with open(save_path, 'w') as f:
        f.write(txt + "\n")
    print(f"  Metrics saved to {save_path}")


def save_entropy_metrics_txt(vol_df, save_path, title, unit="volume"):
    correct   = vol_df.loc[vol_df['correct'] == 1, 'entropy'].values
    incorrect = vol_df.loc[vol_df['correct'] == 0, 'entropy'].values

    def _stats(a):
        if len(a) == 0:
            return (float('nan'),) * 4
        return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), a.min(), a.max()

    c_mean, c_std, c_min, c_max = _stats(correct)
    i_mean, i_std, i_min, i_max = _stats(incorrect)

    lines = [
        title,
        f"Correct/incorrect at Youden threshold; entropy in bits (0 = certain, 1 = max)",
        f"{'-'*66}",
        f"{'Group':<22} {'n':>8} {'mean':>10} {'std':>10} {'min':>7} {'max':>7}",
        f"{'Correct predictions':<22} {len(correct):>8} {c_mean:>10.4f} {c_std:>10.4f} "
        f"{c_min:>7.4f} {c_max:>7.4f}",
        f"{'Incorrect predictions':<22} {len(incorrect):>8} {i_mean:>10.4f} {i_std:>10.4f} "
        f"{i_min:>7.4f} {i_max:>7.4f}",
        f"{'-'*66}",
        f"{'Difference (incorrect - correct)':<40} {i_mean - c_mean:>10.4f}",
        f"{'Ratio (incorrect / correct)':<40} "
        f"{(i_mean / c_mean) if c_mean else float('nan'):>10.4f}",
        f"",
        f"{f'All {unit}s':<22} {len(vol_df):>8} {vol_df['entropy'].mean():>10.4f} "
        f"{vol_df['entropy'].std(ddof=1):>10.4f}",
    ]
    txt = "\n".join(lines)
    print(f"\n{txt}")
    with open(save_path, 'w') as f:
        f.write(txt + "\n")
    print(f"  Entropy metrics saved to {save_path}")
    return c_mean, i_mean


def save_correct_incorrect_plot(vol_df, save_path, title, unit="volume"):
    from scipy.stats import gaussian_kde
    groups = [
        ('Correct predictions',   vol_df.loc[vol_df['correct'] == 1, 'entropy'].values, '#2ca02c'),
        ('Incorrect predictions', vol_df.loc[vol_df['correct'] == 0, 'entropy'].values, '#d62728'),
    ]
    grid = np.linspace(0, 1, 512)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for label, vals, color in groups:
        if len(vals) == 0:
            continue
        try:
            if len(vals) < 2 or np.std(vals) == 0:
                raise ValueError("degenerate group")
            dens = gaussian_kde(vals)(grid)
            ax.plot(grid, dens, color=color, linewidth=2.5,
                    label=f'{label} (n={len(vals)}, mean={vals.mean():.3f})')
            ax.fill_between(grid, dens, color=color, alpha=0.15)
        except Exception:
            counts, edges = np.histogram(vals, bins=np.linspace(0, 1, 31), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, counts, color=color, linewidth=2.5,
                    label=f'{label} (n={len(vals)}, mean={vals.mean():.3f})')
    ax.set_xlim(0, 1)
    ax.set_xlabel(f'{unit.capitalize()}-level entropy (bits)', fontsize=12,
                  fontweight='bold', labelpad=10)
    ax.set_ylabel('Density', fontsize=12, fontweight='bold', labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Correct/incorrect entropy distribution saved to {save_path}")


def save_confusion_matrix(labels, preds, save_path, disease_name, title):
    cm      = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im      = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    names = ['Control', disease_name]
    ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(names, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=12)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration — youden.py's evaluate_run from the point predictions exist
# ══════════════════════════════════════════════════════════════════════════════

def youden_entropy_report(val_df, test_df, out_dir, disease_name, backbone,
                          model_type, unit="volume", id_col="vol_id",
                          extra_summary=None) -> dict:
    """Youden threshold on val -> test metrics + entropy UQ, written to out_dir.

    val_df / test_df need columns [id_col, true_label, prob_disease, age], one
    row per scoring unit (age may be NaN). Returns the summary that is also
    written to youden_threshold.json; extra_summary is merged into it.
    """
    for name, df in (('val', val_df), ('test', test_df)):
        if df['true_label'].nunique() < 2:
            raise ValueError(f"{name} split has a single class at {unit} level; "
                             f"a Youden threshold / AUC needs both")
    os.makedirs(out_dir, exist_ok=True)
    cols    = [id_col, 'true_label', 'prob_disease', 'age']
    val_df  = val_df[cols].copy()
    test_df = test_df[cols].copy()
    val_df['true_label']  = val_df['true_label'].astype(int)
    test_df['true_label'] = test_df['true_label'].astype(int)

    # ── Youden threshold on val ──────────────────────────────────────────────
    yinfo = youden_threshold(val_df['true_label'].values, val_df['prob_disease'].values)
    thr   = yinfo['threshold']
    print(f"\n  Youden threshold (val, {unit}): {thr:.4f}  J={yinfo['youden_j']:.3f}  "
          f"sens={yinfo['val_sensitivity']:.3f}  spec={yinfo['val_specificity']:.3f}")

    val_df['pred_label'] = (val_df['prob_disease'] >= thr).astype(int)
    val_df['correct']    = (val_df['pred_label'] == val_df['true_label']).astype(int)
    val_df.round(4).to_csv(os.path.join(out_dir, f'val_{unit}_predictions.csv'), index=False)

    # ── Apply to test ────────────────────────────────────────────────────────
    labels = test_df['true_label'].values
    probs  = test_df['prob_disease'].values
    test_df['prob_control']  = 1.0 - probs
    test_df['entropy']       = entropy_bits(probs)
    test_df['pred_label_05'] = (probs >= 0.5).astype(int)
    test_df['correct_05']    = (test_df['pred_label_05'] == labels).astype(int)
    test_df['pred_label']    = (probs >= thr).astype(int)
    test_df['correct']       = (test_df['pred_label'] == labels).astype(int)
    test_df = test_df[[id_col, 'true_label', 'prob_control', 'prob_disease', 'entropy',
                       'age', 'pred_label_05', 'correct_05', 'pred_label', 'correct']]
    test_df.round(4).to_csv(os.path.join(out_dir, f'test_{unit}_predictions_youden.csv'),
                            index=False)

    test_auc = float(roc_auc_score(labels, probs))
    m05      = threshold_metrics(labels, probs, 0.5)
    my       = threshold_metrics(labels, probs, thr)

    dn  = disease_name
    tag = f"{dn} ({backbone}, {model_type})"
    save_metrics_txt(test_auc, m05, my, yinfo,
                     os.path.join(out_dir, 'test_metrics_youden.txt'),
                     title=f"{tag} — test, {unit}-level")
    c_mean, i_mean = save_entropy_metrics_txt(
        test_df, os.path.join(out_dir, 'entropy_metrics_youden.txt'),
        title=f"{tag} — Entropy UQ, test, {unit}-level (Youden thr={thr:.4f})",
        unit=unit)
    save_correct_incorrect_plot(
        test_df, os.path.join(out_dir, 'entropy_dist_correct_vs_incorrect_youden.png'),
        title=f"{dn} ({backbone}) — Entropy: Correct vs Incorrect (Youden {thr:.3f})",
        unit=unit)
    save_confusion_matrix(
        labels, test_df['pred_label'].values,
        os.path.join(out_dir, 'confusion_matrix_youden.png'), dn,
        title=f"Confusion Matrix — test, {unit}, Youden {thr:.3f} ({backbone})")

    summary = {
        **(extra_summary or {}),
        'backbone':   backbone,
        'model_type': model_type,
        'unit':       unit,
        **yinfo,
        'n_test':     int(len(labels)),
        'n_test_pos': int((labels == 1).sum()),
        'test_auc': test_auc,
        'test_at_0.5':    m05,
        'test_at_youden': my,
        'test_entropy_mean_correct':   float(c_mean),
        'test_entropy_mean_incorrect': float(i_mean),
    }
    with open(os.path.join(out_dir, 'youden_threshold.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Youden outputs saved to {out_dir}")
    return summary


def print_summary_table(summaries, label_key='backbone'):
    """youden.py's closing table, one row per summary."""
    print(f"\n{'═'*100}")
    print(f"  {label_key:<20} {'thr':>7} {'valJ':>7} {'testAUC':>8} "
          f"{'bacc@.5':>8} {'bacc@Y':>8} {'sens@Y':>7} {'spec@Y':>7} "
          f"{'H_corr':>7} {'H_inc':>7}")
    print(f"  {'-'*96}")
    for s in summaries:
        print(f"  {str(s[label_key])[:20]:<20} {s['threshold']:>7.4f} {s['youden_j']:>7.3f} "
              f"{s['test_auc']:>8.3f} {s['test_at_0.5']['bacc']:>8.3f} "
              f"{s['test_at_youden']['bacc']:>8.3f} "
              f"{s['test_at_youden']['sensitivity']:>7.3f} "
              f"{s['test_at_youden']['specificity']:>7.3f} "
              f"{s['test_entropy_mean_correct']:>7.4f} "
              f"{s['test_entropy_mean_incorrect']:>7.4f}")
    print(f"{'═'*100}")
