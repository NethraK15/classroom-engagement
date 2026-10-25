import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config_binary_transformer as config


# =====================================================================
# SLIDING WINDOW DATASET
# =====================================================================
class SlidingWindowDataset(Dataset):
    """
    Dataset that loads pre-extracted fused sequence features (.pt files)
    and applies a sliding window to produce multiple training windows per video.

    This ensures the model learns from the ENTIRE video (all frames),
    not just the first WINDOW_SIZE frames.

    Window strategy:
      - window_size  : number of frames per window (e.g. 120)
      - window_stride: step between consecutive windows (e.g. 30)
      - For a video with N frames, the number of windows is:
            ceil((N - window_size) / stride) + 1
        with the last window padded if it has fewer than window_size frames.
    """

    def __init__(self, file_paths, window_size=config.WINDOW_SIZE,
                 window_stride=config.WINDOW_STRIDE,
                 global_mean=None, global_std=None):
        self.window_size = window_size
        self.window_stride = window_stride
        self.global_mean = global_mean
        self.global_std = global_std

        # Build index: each entry maps to (file_path, start_frame, actual_len, label)
        self.windows = []
        for fp in file_paths:
            try:
                data = torch.load(fp, map_location="cpu")
                sequence = data["sequence"]       # [N_frames, feat_dim]
                label    = int(data["label"])
                n_frames = sequence.shape[0]

                # Generate sliding window start positions
                start = 0
                while True:
                    end = start + window_size
                    if end > n_frames:
                        # Last (possibly short) window — always include it
                        actual_len = n_frames - start
                        if actual_len > 0:
                            self.windows.append((fp, start, actual_len, label))
                        break
                    else:
                        self.windows.append((fp, start, window_size, label))

                    start += window_stride
                    if start >= n_frames:
                        break

            except Exception as e:
                print(f"[Dataset] Warning: Failed to index '{os.path.basename(fp)}': {e}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        fp, start, actual_len, label = self.windows[idx]

        data = torch.load(fp, map_location="cpu")
        sequence = data["sequence"]   # [N_frames, feat_dim]
        feat_dim = sequence.shape[1]

        # Extract window
        window_seq = sequence[start : start + actual_len]

        # Global feature scaling
        if self.global_mean is not None and self.global_std is not None:
            window_seq = (window_seq - self.global_mean) / self.global_std

        # Pad to window_size if shorter
        if actual_len < self.window_size:
            padding = torch.zeros(
                (self.window_size - actual_len, feat_dim), dtype=torch.float32
            )
            window_seq = torch.cat([window_seq, padding], dim=0)

        return window_seq, actual_len, label


# =====================================================================
# VIDEO-LEVEL INFERENCE DATASET (non-overlapping windows for evaluation)
# =====================================================================
class VideoLevelDataset(Dataset):
    """
    Used at inference time. Returns ALL windows for a video, grouped.
    The evaluate script aggregates window predictions per video.
    """
    def __init__(self, file_paths, window_size=config.WINDOW_SIZE,
                 window_stride=config.WINDOW_STRIDE,
                 global_mean=None, global_std=None):
        self.window_size = window_size
        self.window_stride = window_stride
        self.global_mean = global_mean
        self.global_std = global_std
        self.file_paths = file_paths

        # Build: [(fp, start, actual_len, label), ...]
        self.windows = []
        self.video_map = {}  # fp -> list of (window_idx)
        for fp in file_paths:
            try:
                data = torch.load(fp, map_location="cpu")
                sequence = data["sequence"]
                label    = int(data["label"])
                n_frames = sequence.shape[0]

                self.video_map[fp] = []
                start = 0
                while True:
                    end = start + window_size
                    if end > n_frames:
                        actual_len = n_frames - start
                        if actual_len > 0:
                            self.video_map[fp].append(len(self.windows))
                            self.windows.append((fp, start, actual_len, label))
                        break
                    else:
                        self.video_map[fp].append(len(self.windows))
                        self.windows.append((fp, start, window_size, label))

                    start += window_stride
                    if start >= n_frames:
                        break

            except Exception as e:
                print(f"[Dataset] Warning: Could not index '{os.path.basename(fp)}': {e}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        fp, start, actual_len, label = self.windows[idx]

        data = torch.load(fp, map_location="cpu")
        sequence = data["sequence"]
        feat_dim = sequence.shape[1]

        window_seq = sequence[start : start + actual_len]
        if self.global_mean is not None and self.global_std is not None:
            window_seq = (window_seq - self.global_mean) / self.global_std

        if actual_len < self.window_size:
            padding = torch.zeros(
                (self.window_size - actual_len, feat_dim), dtype=torch.float32
            )
            window_seq = torch.cat([window_seq, padding], dim=0)

        return window_seq, actual_len, label


# =====================================================================
# GLOBAL STATS  (computed on training windows only)
# =====================================================================
def compute_global_stats(file_paths):
    """Compute feature mean & std across ALL frames in the training set."""
    all_frames = []
    for f in file_paths:
        try:
            data = torch.load(f, map_location="cpu")
            all_frames.append(data["sequence"])
        except Exception:
            continue
    if not all_frames:
        return None, None
    concat = torch.cat(all_frames, dim=0)
    mean = concat.mean(dim=0, keepdim=True)
    std  = concat.std(dim=0,  keepdim=True) + 1e-6
    return mean, std


# =====================================================================
# DATALOADER BUILDER
# =====================================================================
def get_binary_dataloaders(
    sequence_fused_dir,
    batch_size=config.BATCH_SIZE,
    random_state=config.RANDOM_STATE
):
    """
    Loads binary sequence feature files (engaged / disengaged only).
    Completely skips any file labelled as 'partially engaged'.

    Returns train / val / test DataLoaders using sliding windows.
    """
    if not os.path.exists(sequence_fused_dir):
        raise FileNotFoundError(
            f"Sequence feature folder not found: {sequence_fused_dir}. "
            "Run preprocess/fuse_sequence_features_binary.py first."
        )

    all_files = [
        os.path.join(sequence_fused_dir, f)
        for f in os.listdir(sequence_fused_dir)
        if f.endswith(".pt")
    ]

    if not all_files:
        raise ValueError(
            f"No .pt sequence files found in {sequence_fused_dir}. "
            "Run preprocess/fuse_sequence_features_binary.py first."
        )

    # ------------------------------------------------------------------
    # Filter: keep ONLY engaged (label=0) and disengaged (label=1)
    # Partially engaged files will have label=None or a different label —
    # they are simply skipped here regardless.
    # ------------------------------------------------------------------
    VALID_LABELS = {0, 1}  # 0 = engaged, 1 = disengaged

    labels        = []
    filtered_files = []

    # Determine expected feature dim from valid files
    dims = []
    for f in all_files[:min(20, len(all_files))]:
        try:
            d = torch.load(f, map_location="cpu")
            lbl = int(d["label"])
            if lbl in VALID_LABELS:
                dims.append(d["sequence"].shape[1])
        except Exception:
            continue
    expected_dim = max(set(dims), key=dims.count) if dims else None

    skipped_partial = 0
    for f in all_files:
        try:
            d       = torch.load(f, map_location="cpu")
            lbl     = int(d["label"])
            feat_dim = d["sequence"].shape[1]

            if lbl not in VALID_LABELS:
                skipped_partial += 1
                continue

            if expected_dim is not None and feat_dim != expected_dim:
                print(f"[DataLoader] Skipping dim-mismatch file '{os.path.basename(f)}'.")
                continue

            labels.append(lbl)
            filtered_files.append(f)
        except Exception as e:
            print(f"[DataLoader] Warning: Failed to load '{os.path.basename(f)}': {e}. Skipping.")

    if skipped_partial > 0:
        print(f"[DataLoader] Skipped {skipped_partial} partially-engaged files (not used in binary training).")

    print(f"[DataLoader] Binary dataset: {len(filtered_files)} files "
          f"(engaged: {labels.count(0)}, disengaged: {labels.count(1)})")

    if len(filtered_files) < 3:
        raise ValueError("Not enough valid binary samples to split into train/val/test.")

    # Stratified split: 70% train, 15% val, 15% test
    try:
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            filtered_files, labels, test_size=0.30, stratify=labels, random_state=random_state
        )
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.50, stratify=temp_labels, random_state=random_state
        )
        print("[DataLoader] Stratified split successful.")
    except ValueError as e:
        print(f"[DataLoader] WARNING: Stratified split failed ({e}). Falling back to random split.")
        train_files, temp_files = train_test_split(filtered_files, test_size=0.30, random_state=random_state)
        val_files, test_files   = train_test_split(temp_files, test_size=0.50, random_state=random_state)

    # Global stats on training files only
    global_mean, global_std = compute_global_stats(train_files)
    print(f"[DataLoader] Global stats computed on {len(train_files)} training videos.")

    # Sliding-window datasets
    train_ds = SlidingWindowDataset(train_files, global_mean=global_mean, global_std=global_std)
    val_ds   = SlidingWindowDataset(val_files,   global_mean=global_mean, global_std=global_std)
    test_ds  = VideoLevelDataset(test_files,     global_mean=global_mean, global_std=global_std)

    num_workers = 0 if os.name == "nt" else 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[DataLoader] Sliding-window splits:")
    print(f"             - Train videos: {len(train_files)}  → {len(train_ds)} windows")
    print(f"             - Val   videos: {len(val_files)}  → {len(val_ds)} windows")
    print(f"             - Test  videos: {len(test_files)}  → {len(test_ds)} windows")

    return train_loader, val_loader, test_loader, test_files, global_mean, global_std
