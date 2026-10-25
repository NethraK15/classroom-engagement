import os
import sys
import json
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import config_binary_transformer as config
from data_loader_binary_transformer import get_binary_dataloaders, VideoLevelDataset
from models_binary_transformer import BinaryEngagementTransformer


# =====================================================================
# VIDEO-LEVEL PREDICTION (window aggregation)
# =====================================================================
def predict_video_level(model, test_files, global_mean, global_std, device):
    """
    For each test video:
      1. Generate all sliding windows.
      2. Run transformer on each window.
      3. Average the softmax probabilities (soft voting) across all windows.
      4. Final label = argmax of averaged probabilities.

    This gives a SINGLE prediction per video, using the entire video content.
    """
    model.eval()

    video_targets = []
    video_preds   = []
    video_probs   = []
    video_names   = []

    with torch.no_grad():
        for fp in test_files:
            try:
                data     = torch.load(fp, map_location="cpu")
                sequence = data["sequence"]   # [N, feat_dim]
                label    = int(data["label"])
                n_frames = sequence.shape[0]
                feat_dim = sequence.shape[1]

                # Normalise
                if global_mean is not None and global_std is not None:
                    mean = global_mean.to(sequence.device)
                    std = global_std.to(sequence.device)
                    sequence = (sequence - mean) / std

                # Generate windows
                window_probs_list = []
                start = 0
                while True:
                    end        = start + config.WINDOW_SIZE
                    actual_len = min(config.WINDOW_SIZE, n_frames - start)

                    if actual_len <= 0:
                        break

                    window = sequence[start: start + actual_len]

                    # Pad if needed
                    if actual_len < config.WINDOW_SIZE:
                        pad = torch.zeros(
                            (config.WINDOW_SIZE - actual_len, feat_dim), dtype=torch.float32
                        )
                        window = torch.cat([window, pad], dim=0)

                    window_tensor = window.unsqueeze(0).to(device)   # [1, T, D]
                    seq_len_t     = torch.tensor([actual_len], device=device)

                    logits = model(window_tensor, seq_len_t)
                    probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    window_probs_list.append(probs)

                    start += config.WINDOW_STRIDE
                    if start >= n_frames:
                        break

                if not window_probs_list:
                    continue

                # Soft-vote aggregation
                avg_probs  = np.mean(window_probs_list, axis=0)   # [2]
                pred_label = int(np.argmax(avg_probs))

                video_targets.append(label)
                video_preds.append(pred_label)
                video_probs.append(avg_probs)
                video_names.append(os.path.basename(fp))

            except Exception as e:
                print(f"[Evaluate] Warning: Could not predict '{os.path.basename(fp)}': {e}")

    return (
        np.array(video_targets),
        np.array(video_preds),
        np.array(video_probs),
        video_names
    )


# =====================================================================
# PLOTS
# =====================================================================
def save_confusion_matrix(targets, preds, class_names, save_path):
    labels_range = list(range(len(class_names)))
    cm = confusion_matrix(targets, preds, labels=labels_range)
    plt.figure(figsize=(7, 6))
    if HAS_SEABORN:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 14}
        )
    else:
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.title("Binary Engagement Transformer — Confusion Matrix", fontsize=13, pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Confusion matrix saved: {save_path}")


def save_roc_curve(targets, probs, class_names, save_path):
    """Binary ROC — one curve for the positive class (Disengaged = 1)."""
    plt.figure(figsize=(7, 6))
    colors = ["#2ca02c", "#d62728"]  # Engaged / Disengaged

    # One-vs-rest for each class
    targets_bin = label_binarize(targets, classes=list(range(len(class_names))))
    if targets_bin.shape[1] == 1:
        targets_bin = np.hstack([1 - targets_bin, targets_bin])

    for i, cls_name in enumerate(class_names):
        if len(np.unique(targets_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(targets_bin[:, i], probs[:, i])
            roc_auc     = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                     label=f"{cls_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Binary Engagement Transformer — ROC Curves", fontsize=13, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluate] ROC curve saved: {save_path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 60)
    print("  EVALUATING BINARY ENGAGEMENT TRANSFORMER")
    print("  (Video-level predictions via sliding window aggregation)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(config.MODEL_WEIGHTS_PATH):
        print(f"[Error] Weights not found: {config.MODEL_WEIGHTS_PATH}")
        print("Please run train_binary_transformer.py first.")
        sys.exit(1)

    checkpoint = torch.load(config.MODEL_WEIGHTS_PATH, map_location=device)

    input_dim    = checkpoint.get("input_dim", 1989)
    d_model      = checkpoint.get("d_model",   config.D_MODEL)
    nhead        = checkpoint.get("nhead",      config.NHEAD)
    num_layers   = checkpoint.get("num_layers", config.NUM_LAYERS)
    num_classes  = checkpoint.get("num_classes", config.NUM_CLASSES)
    global_mean  = checkpoint.get("global_mean", None)
    global_std   = checkpoint.get("global_std",  None)

    # Get test split (same random state ensures reproducibility)
    _, _, _, test_files, gm, gs = get_binary_dataloaders(
        sequence_fused_dir=config.SEQUENCE_FUSED_DIR,
        batch_size=config.BATCH_SIZE,
        random_state=config.RANDOM_STATE
    )
    if global_mean is None:
        global_mean, global_std = gm, gs

    model = BinaryEngagementTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint — input_dim: {input_dim}, d_model: {d_model}\n")

    # ── Video-level prediction ────────────────────────────────────────
    targets, preds, probs, video_names = predict_video_level(
        model, test_files, global_mean, global_std, device
    )

    if len(targets) == 0:
        raise RuntimeError(
            "[Evaluate] No video predictions were produced. "
            "Check the per-video warnings above for data or device errors."
        )

    # ── Metrics ──────────────────────────────────────────────────────
    acc                     = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="weighted", zero_division=0
    )

    print("=" * 60)
    print("  BINARY TRANSFORMER — VIDEO-LEVEL METRICS")
    print("=" * 60)
    print(f"  Accuracy   : {acc * 100:6.2f}%")
    print(f"  Precision  : {precision * 100:6.2f}%  (weighted)")
    print(f"  Recall     : {recall * 100:6.2f}%  (weighted)")
    print(f"  F1-score   : {f1 * 100:6.2f}%  (weighted)")
    print("=" * 60)

    labels_range = list(range(len(config.CLASS_NAMES)))
    report_text = classification_report(
        targets, preds,
        labels=labels_range,
        target_names=config.CLASS_NAMES,
        zero_division=0
    )
    print("\nClassification Report:")
    print(report_text)

    # ── Save outputs ─────────────────────────────────────────────────
    os.makedirs(config.EVAL_OUT_DIR, exist_ok=True)

    metrics_data = {
        "model"              : "Binary Engagement Transformer",
        "classes"            : config.CLASS_NAMES,
        "accuracy"           : float(acc),
        "precision_weighted" : float(precision),
        "recall_weighted"    : float(recall),
        "f1_weighted"        : float(f1),
        "classification_report": classification_report(
            targets, preds,
            labels=labels_range,
            target_names=config.CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )
    }
    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[Evaluate] Metrics JSON saved: {config.METRICS_JSON_PATH}")

    save_confusion_matrix(targets, preds, config.CLASS_NAMES, config.CONFUSION_MATRIX_PATH)
    save_roc_curve(targets, probs, config.CLASS_NAMES, config.ROC_CURVE_PATH)

    # Predictions CSV
    df = pd.DataFrame({
        "video_file"      : video_names,
        "true_label_idx"  : targets,
        "true_label"      : [config.CLASS_NAMES[t] for t in targets],
        "pred_label_idx"  : preds,
        "pred_label"      : [config.CLASS_NAMES[p] for p in preds],
        "prob_engaged"    : probs[:, 0],
        "prob_disengaged" : probs[:, 1],
        "correct"         : (targets == preds).astype(int)
    })
    df.to_csv(config.PREDICTIONS_CSV_PATH, index=False)
    print(f"[Evaluate] Predictions CSV saved: {config.PREDICTIONS_CSV_PATH}")

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
