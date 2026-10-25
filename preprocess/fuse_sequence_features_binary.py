"""
fuse_sequence_features_binary.py
─────────────────────────────────
COMPLETE ALL-IN-ONE PREPROCESSING PIPELINE for Binary Engagement Transformer.

  Step 1: Scan dataset — ONLY engaged & disengaged (partial IGNORED)
  Step 2: Extract ALL raw frames from each video & store in Google Drive
  Step 3: Extract EfficientNet-B0 CNN features  → [N, 1280] per video
  Step 4: Extract OpenFace facial features       → [N, 709]  per video
           *** OpenFace is REQUIRED. Script exits if not found. ***
           *** Run the OpenFace installation cell in Colab first! ***
  Step 5: Fuse CNN + OpenFace → [N, 1989] full temporal sequence → .pt

Feature dim: 1280 (CNN) + 709 (OpenFace) = 1989 per frame
Classes:     engaged (0)  |  disengaged (1)
Skipped:     partially engaged (completely ignored)
"""

import os
import sys
import subprocess
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_binary_transformer as config

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.3gp', '.flv', '.webm')
IMAGE_SIZE       = (224, 224)
CNN_BATCH_SIZE   = 32


# ─────────────────────────────────────────────────────────────────────
# STEP 1 — FIND BINARY VIDEO FILES (skip partial)
# ─────────────────────────────────────────────────────────────────────
def find_binary_videos(dataset_dir):
    video_list      = []
    matched_folders = {0: [], 1: []}
    skipped_partial = []

    if not os.path.exists(dataset_dir):
        print(f"[Error] Dataset directory not found: {dataset_dir}")
        return video_list

    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if not os.path.isdir(item_path):
            continue
        norm = item.lower().replace(" ", "").replace("_", "").replace("-", "")
        if "partial" in norm:
            skipped_partial.append(item)
            continue
        if "disengaged" in norm:
            matched_folders[1].append(item)
        elif "engaged" in norm:
            matched_folders[0].append(item)

    if skipped_partial:
        print(f"[Step 1] Skipped partially-engaged folder(s): {skipped_partial}")

    class_names = config.CLASSES  # ["engaged", "disengaged"]
    for label_idx in range(2):
        cls = class_names[label_idx]
        for folder in matched_folders[label_idx]:
            print(f"[Step 1] '{folder}' → class '{cls}' (label {label_idx})")
            class_dir = os.path.join(dataset_dir, folder)
            for root, _, files in os.walk(class_dir):
                for fname in sorted(files):
                    if fname.lower().endswith(VIDEO_EXTENSIONS):
                        video_list.append({
                            "video_path" : os.path.join(root, fname),
                            "class_name" : cls,
                            "label"      : label_idx,
                            "filename"   : fname,
                            "base_name"  : os.path.splitext(fname)[0]
                        })
    return video_list


# ─────────────────────────────────────────────────────────────────────
# STEP 2 — EXTRACT ALL RAW FRAMES TO GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────────────
def extract_and_store_frames_to_drive(video_path, frames_out_dir, target_size=IMAGE_SIZE):
    """
    Extracts ALL frames from video and saves them to Google Drive (frames_out_dir).
    If frames are already extracted on Drive, loads them directly from Drive.

    Returns:
        np.ndarray: [N, H, W, 3] uint8 RGB frames array (or None on failure).
    """
    os.makedirs(frames_out_dir, exist_ok=True)

    # Check if frames already exist on Google Drive (resume-safe)
    valid_exts = ('.jpg', '.jpeg', '.png')
    existing_files = sorted([
        f for f in os.listdir(frames_out_dir)
        if f.lower().endswith(valid_exts) and not f.startswith("._")
    ])

    if len(existing_files) > 0:
        frames = []
        for f in existing_files:
            img_bgr = cv2.imread(os.path.join(frames_out_dir, f))
            if img_bgr is not None:
                if target_size is not None and (img_bgr.shape[1], img_bgr.shape[0]) != target_size:
                    img_bgr = cv2.resize(img_bgr, target_size)
                frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if frames:
            return np.array(frames, dtype=np.uint8)

    # Otherwise extract from video and save each frame directly to Google Drive
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        save_img = cv2.resize(frame, target_size) if target_size is not None else frame
        frame_filename = f"frame_{frame_idx:05d}.jpg"
        frame_filepath = os.path.join(frames_out_dir, frame_filename)
        cv2.imwrite(frame_filepath, save_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        frames.append(cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
        frame_idx += 1

    cap.release()
    return np.array(frames, dtype=np.uint8) if frames else None


def extract_frames_from_video(video_path, target_size=IMAGE_SIZE, frames_out_dir=None):
    """
    Extracts frames from video. If frames_out_dir is provided, saves/loads
    frames to/from Google Drive.
    """
    if frames_out_dir is not None:
        return extract_and_store_frames_to_drive(video_path, frames_out_dir, target_size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, target_size), cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames, dtype=np.uint8) if frames else None


# ─────────────────────────────────────────────────────────────────────
# STEP 3 — EFFICIENTNET-B0 CNN FEATURES
# ─────────────────────────────────────────────────────────────────────
def build_efficientnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=True)
    model.classifier = nn.Identity()   # 1280-dim output
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
    return model, normalize, device


def extract_cnn_features(frames_np, model, normalize, device, batch_size=CNN_BATCH_SIZE):
    """frames_np [N,H,W,3] → [N, 1280] float32"""
    tensor    = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 255.0
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = torch.stack([normalize(img) for img in tensor[i:i+batch_size]]).to(device)
            all_feats.append(model(batch).cpu().numpy())
    return np.concatenate(all_feats, axis=0)   # [N, 1280]


# ─────────────────────────────────────────────────────────────────────
# STEP 4 — OPENFACE FEATURES  (REQUIRED — no fallback)
# ─────────────────────────────────────────────────────────────────────
OPENFACE_SEARCH_PATHS = [
    "/content/OpenFace/build/bin/FeatureExtraction",
    "/content/OpenFace/exe/FeatureExtraction/FeatureExtraction",
    "/content/OpenFace/exe/FeatureExtraction",
    "FeatureExtraction",
    "d:/Amrita/Final_Year_Project/Sem_6 Work/OpenFace_2.2.0_win_x64/FeatureExtraction.exe",
]

def find_openface_executable():
    """Return path to OpenFace FeatureExtraction binary, or None."""
    for path in OPENFACE_SEARCH_PATHS:
        if path == "FeatureExtraction":
            try:
                ret = subprocess.call([path, "--help"],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
                return path
            except FileNotFoundError:
                continue
        if os.path.isfile(path):
            try:
                os.chmod(path, 0o755)
            except Exception:
                pass
            return path
    return None


def require_openface():
    """
    Locate OpenFace. If not found, print clear instructions and EXIT.
    No fallback — OpenFace is mandatory.
    """
    exe = find_openface_executable()
    if exe:
        return exe

    print("\n" + "=" * 65)
    print("  ERROR: OpenFace FeatureExtraction NOT FOUND")
    print("=" * 65)
    print("\n  OpenFace is required for this pipeline.")
    print("  Run the following installation cell in your Colab notebook")
    print("  BEFORE running this preprocessing script:\n")
    print("  ── Colab Installation Cell ──────────────────────────────")
    print("  !apt-get update -qq && apt-get install -y --fix-missing build-essential \\")
    print("       cmake libopenblas-dev liblapack-dev libx11-dev libgtk2.0-dev \\")
    print("       libboost-all-dev libdlib-dev libtbb-dev libopencv-dev")
    print("  !git clone --depth=1 https://github.com/TadasBaltrusaitis/OpenFace.git /content/OpenFace")
    print("  !bash /content/OpenFace/download_models.sh")
    print("  !mkdir -p /content/OpenFace/build")
    print("  !cd /content/OpenFace/build && cmake -D CMAKE_BUILD_TYPE=RELEASE .. && make -j$(nproc)")
    print("  ─────────────────────────────────────────────────────────")
    print("\n  After installation verify with:")
    print("  !ls /content/OpenFace/build/bin/FeatureExtraction")
    print("=" * 65 + "\n")
    sys.exit(1)


def extract_openface_features(video_path, n_frames, openface_exe, temp_dir):
    """
    Run OpenFace on video, parse CSV, return [N, feat_dim] float32.
    Raises RuntimeError on failure — caller treats it as a hard error.
    """
    import pandas as pd

    os.makedirs(temp_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    cmd = [openface_exe, "-f", video_path, "-out_dir", temp_dir]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"OpenFace exited with code {result.returncode}: "
            f"{result.stderr.decode('utf-8', errors='ignore')[:200]}"
        )

    # Locate generated CSV
    csv_path = os.path.join(temp_dir, f"{video_stem}.csv")
    if not os.path.exists(csv_path):
        for f in os.listdir(temp_dir):
            if f.lower().endswith(".csv") and video_stem.lower() in f.lower():
                csv_path = os.path.join(temp_dir, f)
                break

    if not os.path.exists(csv_path):
        raise RuntimeError(f"OpenFace ran but no CSV found for '{video_stem}'")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "face_id" in df.columns:
        df = df[df["face_id"] == 0]

    df = df.set_index("frame")
    df = df.reindex(pd.Index(range(1, n_frames + 1), name="frame"))

    meta_cols = ["face_id", "timestamp", "confidence", "success"]
    df = df.drop(columns=[c for c in meta_cols if c in df.columns])
    df = df.fillna(0.0)

    feats = df.values.astype(np.float32)
    if feats.shape[1] == 0:
        raise RuntimeError(f"OpenFace CSV for '{video_stem}' has zero feature columns")

    return feats   # [N, openface_dim]


# ─────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  BINARY TRANSFORMER — COMPLETE PREPROCESSING PIPELINE")
    print("  Classes: Engaged (0)  |  Disengaged (1)")
    print("  Partially Engaged: COMPLETELY SKIPPED")
    print("  OpenFace: REQUIRED (no fallback)")
    print("=" * 65)
    print(f"Dataset dir       : {config.DATASET_DIR}")
    print(f"Extracted frames  : {config.EXTRACTED_FRAMES_DIR} (Google Drive)")
    print(f"Output dir        : {config.SEQUENCE_FUSED_DIR}")
    print(f"Window size       : {config.WINDOW_SIZE} frames  (stride {config.WINDOW_STRIDE})")
    print()

    config.ensure_output_dirs()
    os.makedirs(config.EXTRACTED_FRAMES_DIR, exist_ok=True)
    os.makedirs(config.SEQUENCE_FUSED_DIR, exist_ok=True)

    # ── Step 1: Find binary videos ─────────────────────────────────
    videos = find_binary_videos(config.DATASET_DIR)
    if not videos:
        print("[Error] No binary videos found. Check DATASET_DIR in config.")
        sys.exit(1)

    e_count = sum(1 for v in videos if v["label"] == 0)
    d_count = sum(1 for v in videos if v["label"] == 1)
    print(f"\n[Step 1] Found {len(videos)} binary videos "
          f"(engaged: {e_count}, disengaged: {d_count})")

    # ── Step 2: Build EfficientNet ─────────────────────────────────
    print("\n[Step 2] Loading EfficientNet-B0...")
    cnn_model, cnn_normalize, cnn_device = build_efficientnet()
    print(f"         Device: {cnn_device}")

    # ── Step 3: REQUIRE OpenFace — exit if missing ─────────────────
    print("\n[Step 3] Locating OpenFace FeatureExtraction binary...")
    openface_exe = require_openface()   # exits with instructions if not found
    print(f"         Found: {openface_exe}")
    print("         Mode : CNN (1280) + OpenFace (709) = 1989-dim per frame")

    of_temp_dir = os.path.join(config.SEQUENCE_FUSED_DIR, "_openface_tmp")
    os.makedirs(of_temp_dir, exist_ok=True)

    # ── Process each video ─────────────────────────────────────────
    success_count = 0
    skip_count    = 0
    error_count   = 0

    print(f"\n[Step 4] Processing {len(videos)} videos...\n")

    for video in tqdm(videos, desc="Preprocessing"):
        base_name  = video["base_name"]
        class_name = video["class_name"]
        label      = video["label"]
        video_path = video["video_path"]

        save_path = os.path.join(
            config.SEQUENCE_FUSED_DIR, f"{class_name}__{base_name}.pt"
        )
        frames_out_dir = os.path.join(
            config.EXTRACTED_FRAMES_DIR, class_name, base_name
        )

        if os.path.exists(save_path):
            # If features already exist, ensure extracted frames are also safely on Drive
            if not os.path.exists(frames_out_dir) or len(os.listdir(frames_out_dir)) == 0:
                try:
                    extract_and_store_frames_to_drive(video_path, frames_out_dir, target_size=IMAGE_SIZE)
                except Exception:
                    pass
            skip_count += 1
            continue

        try:
            # a. Extract ALL frames & store directly to Google Drive
            frames_np = extract_and_store_frames_to_drive(
                video_path, frames_out_dir, target_size=IMAGE_SIZE
            )
            if frames_np is None or len(frames_np) == 0:
                raise RuntimeError("0 frames extracted")
            n_frames = len(frames_np)

            # b. EfficientNet CNN features  [N, 1280]
            cnn_feats = extract_cnn_features(
                frames_np, cnn_model, cnn_normalize, cnn_device
            )

            # c. OpenFace features  [N, 709]
            of_feats = extract_openface_features(
                video_path, n_frames, openface_exe, of_temp_dir
            )

            # d. Align frames & fuse  [N, 1989]
            n_aligned = min(len(cnn_feats), len(of_feats))
            fused = np.concatenate(
                [cnn_feats[:n_aligned], of_feats[:n_aligned]], axis=1
            ).astype(np.float32)

            # e. Save full sequence
            torch.save({
                "sequence"    : torch.from_numpy(fused).float(),
                "seq_len"     : n_aligned,
                "label"       : label,
                "video_name"  : video["filename"],
                "class_name"  : class_name,
                "feature_dim" : fused.shape[1]   # should be 1989
            }, save_path)

            success_count += 1

        except Exception as e:
            tqdm.write(f"\n  [Error] Failed '{base_name}': {e}")
            error_count += 1

    # ── Cleanup temp OpenFace dir ──────────────────────────────────
    try:
        shutil.rmtree(of_temp_dir)
    except Exception:
        pass

    # ── Summary ────────────────────────────────────────────────────
    saved_files = [f for f in os.listdir(config.SEQUENCE_FUSED_DIR) if f.endswith(".pt")]
    print("\n" + "=" * 65)
    print("  PREPROCESSING COMPLETE")
    print("=" * 65)
    print(f"  Mode                   : CNN (1280) + OpenFace (709) = 1989-dim")
    print(f"  Extracted Frames (Drive): {config.EXTRACTED_FRAMES_DIR}")
    print(f"  Successfully processed : {success_count}")
    print(f"  Resumed / Skipped      : {skip_count}")
    print(f"  Failed                 : {error_count}")
    print(f"  Total .pt files saved  : {len(saved_files)}")

    if error_count > 0:
        print(f"\n  [Warning] {error_count} video(s) failed. Check errors above.")
        print("  You can safely re-run this script — processed files are skipped.")

    print("\n  Feature dim : 1989 per frame")
    print(f"  Sliding windows (size={config.WINDOW_SIZE}, stride={config.WINDOW_STRIDE})")
    print("  are generated ON-THE-FLY in the data loader — not stored here.")
    print("=" * 65)


if __name__ == "__main__":
    main()
