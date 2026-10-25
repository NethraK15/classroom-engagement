import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

import config_binary_transformer as config
from data_loader_binary_transformer import get_binary_dataloaders
from models_binary_transformer import BinaryEngagementTransformer


def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=config.MAX_GRAD_NORM):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for sequences, seq_lens, labels in dataloader:
        sequences = sequences.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences, seq_lens)
        loss = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    with torch.no_grad():
        for sequences, seq_lens, labels in dataloader:
            sequences = sequences.to(device)
            labels    = labels.to(device)

            outputs = model(sequences, seq_lens)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * sequences.size(0)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main():
    print("=" * 60)
    print("  BINARY ENGAGEMENT TRANSFORMER (Engaged vs Disengaged)")
    print("  Sliding Window — Full Video Coverage")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load dataloaders ─────────────────────────────────────────────
    try:
        train_loader, val_loader, _, _, global_mean, global_std = get_binary_dataloaders(
            sequence_fused_dir=config.SEQUENCE_FUSED_DIR,
            batch_size=config.BATCH_SIZE,
            random_state=config.RANDOM_STATE
        )
    except Exception as e:
        print(f"\n[Error] Failed to load sequence data: {e}")
        sys.exit(1)

    # ── Detect input feature dimension ───────────────────────────────
    seq_files = [f for f in os.listdir(config.SEQUENCE_FUSED_DIR) if f.endswith(".pt")]
    if not seq_files:
        print("[Error] Sequence fused features folder is empty.")
        sys.exit(1)

    sample = torch.load(
        os.path.join(config.SEQUENCE_FUSED_DIR, seq_files[0]),
        map_location="cpu"
    )
    input_dim = sample["sequence"].shape[1]

    print(f"Feature Dimension  : {input_dim}")
    print(f"d_model            : {config.D_MODEL}")
    print(f"Attention Heads    : {config.NHEAD}")
    print(f"Transformer Layers : {config.NUM_LAYERS}")
    print(f"Feedforward Dim    : {config.DIM_FEEDFORWARD}")
    print(f"Window Size        : {config.WINDOW_SIZE} frames")
    print(f"Window Stride      : {config.WINDOW_STRIDE} frames")
    print(f"Output Classes     : {config.NUM_CLASSES} (Engaged / Disengaged)\n")

    # ── Model ────────────────────────────────────────────────────────
    model = BinaryEngagementTransformer(
        input_dim=input_dim,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )

    os.makedirs(os.path.dirname(config.MODEL_WEIGHTS_PATH), exist_ok=True)

    # ── Training Loop (Runs All Epochs) ──────────────────────────────
    best_val_acc  = 0.0
    best_val_loss = float("inf")

    print("\nStarting Binary Transformer training (sliding windows)...\n")
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        scheduler.step()

        improved = val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )

        print(
            f"Epoch [{epoch+1:03d}/{config.EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:5.2f}%"
            + (" ✓ Best" if improved else "")
        )

        if improved:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            torch.save({
                "epoch"             : epoch + 1,
                "model_state_dict"  : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc"           : val_acc,
                "val_loss"          : val_loss,
                "input_dim"         : input_dim,
                "d_model"           : config.D_MODEL,
                "nhead"             : config.NHEAD,
                "num_layers"        : config.NUM_LAYERS,
                "window_size"       : config.WINDOW_SIZE,
                "window_stride"     : config.WINDOW_STRIDE,
                "num_classes"       : config.NUM_CLASSES,
                "global_mean"       : global_mean,
                "global_std"        : global_std,
            }, config.MODEL_WEIGHTS_PATH)

    print("\n" + "=" * 60)
    print("       BINARY TRANSFORMER TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val Accuracy : {best_val_acc * 100:.2f}%")
    print(f"Weights saved to  : {config.MODEL_WEIGHTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
