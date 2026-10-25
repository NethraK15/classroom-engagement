"""
test_classroom_binary_transformer.py
─────────────────────────────────────────────────────────────────────────────
CLASSROOM VIDEO INFERENCE PIPELINE FOR BINARY ENGAGEMENT TRANSFORMER
Classes:
  - 0: Engaged     --> GREEN Bounding Box  (RGB: 0, 255, 0)
  - 1: Disengaged  --> RED Bounding Box    (RGB: 0, 0, 255)

Features:
  1. Auto-discovers classroom video in 'clasroom_recording' / 'classroom_recording'
     folder on Google Drive or local workspace if --video is not specified.
  2. Multi-student face/person detection using YOLO + ByteTrack tracking.
  3. EfficientNet-B0 visual feature extraction + dynamic dimension padding.
  4. Temporal Transformer sequence classification (BinaryEngagementTransformer).
  5. Outputs:
     - Annotated Classroom Video (.mp4) with Green/Red boxes + Top HUD overlay.
     - Frame-by-frame Detailed CSV Log.
     - Per-Student Engagement Text Report.
     - Visual Timeline & Student Engagement Analytics Plot (.png).
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")  # Headless mode for server / Colab compatibility
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add current directory and parent to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config_binary_transformer as config
from models_binary_transformer import BinaryEngagementTransformer
from feature_extractor_binary import FusedFeaturePipeline

# Optional imports for YOLO
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. VISUAL FEATURE EXTRACTOR (EfficientNet-B0)
# ─────────────────────────────────────────────────────────────────────────────
class EfficientNetFeatureExtractor:
    """Extracts 1280-dim deep visual features from 224x224 RGB face crops."""
    def __init__(self, device):
        self.device = device
        from torchvision import models
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            model = models.efficientnet_b0(pretrained=True)

        model.classifier = nn.Identity()
        model.eval()
        model.to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        self.model = model

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def extract_crop_feature(self, crop_bgr):
        """Extracts [1280] float32 feature from BGR crop."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        tensor = self.normalize(tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        return feat.squeeze(0).cpu().numpy()  # [1280]


# ─────────────────────────────────────────────────────────────────────────────
# 3. AUTO-DISCOVERY OF CLASSROOM VIDEO
# ─────────────────────────────────────────────────────────────────────────────
def find_classroom_video(explicit_video_path=None, explicit_dir=None):
    """
    Finds a classroom video file. If explicit_video_path is given and valid,
    returns it. Otherwise searches standard 'clasroom_recording' / 'classroom_recording'
    folders on Google Drive or local workspace.
    """
    if explicit_video_path and os.path.exists(explicit_video_path):
        return explicit_video_path

    valid_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    candidates_dirs = []

    if explicit_dir and os.path.exists(explicit_dir):
        candidates_dirs.append(explicit_dir)

    # Search candidates on Drive and local paths
    search_roots = [
        "/content/drive/MyDrive/Student_Engagement_DL",
        "/content/drive/MyDrive",
        "/content",
        config.PROJECT_ROOT,
        os.getcwd(),
        os.path.dirname(os.getcwd()),
        "c:/FYP",
        "d:/Amrita/Final_Year_Project/Sem_6 Work/Student_Engagement_DL"
    ]

    folder_names = [
        "clasroom_recording",
        "classroom_recording",
        "clasroom_recordings",
        "classroom_recordings",
        "Classroom_Recording",
        "Classroom_recordings"
    ]

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for fname in folder_names:
            p = os.path.join(root, fname)
            if os.path.isdir(p) and p not in candidates_dirs:
                candidates_dirs.append(p)
        # Also check direct subdirectories with 'clas' and 'record'
        try:
            for item in os.listdir(root):
                full_p = os.path.join(root, item)
                if os.path.isdir(full_p):
                    norm = item.lower().replace(" ", "").replace("_", "")
                    if "clas" in norm and "record" in norm and full_p not in candidates_dirs:
                        candidates_dirs.append(full_p)
        except Exception:
            pass

    for d in candidates_dirs:
        try:
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if f.lower().endswith(valid_exts) and not f.startswith("._"):
                        found_path = os.path.join(root, f)
                        print(f"[+] Found classroom recording in folder: '{d}'")
                        print(f"    -> Selected video: {found_path}")
                        return found_path
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD TRAINED BINARY TRANSFORMER MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_binary_transformer(model_path, device):
    """
    Loads BinaryEngagementTransformer checkpoint with dynamic parameter detection.
    """
    target_path = model_path if (model_path and os.path.exists(model_path)) else config.MODEL_WEIGHTS_PATH

    input_dim = 1989
    d_model = config.D_MODEL
    nhead = config.NHEAD
    num_layers = config.NUM_LAYERS
    dim_feedforward = config.DIM_FEEDFORWARD
    num_classes = config.NUM_CLASSES
    global_mean = None
    global_std = None
    state_dict = None

    if target_path and os.path.exists(target_path):
        print(f"[+] Loading trained Binary Transformer model from: {target_path}")
        checkpoint = torch.load(target_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            input_dim = checkpoint.get("input_dim", 1989)
            d_model = checkpoint.get("d_model", config.D_MODEL)
            nhead = checkpoint.get("nhead", config.NHEAD)
            num_layers = checkpoint.get("num_layers", config.NUM_LAYERS)
            dim_feedforward = checkpoint.get("dim_feedforward", config.DIM_FEEDFORWARD)
            num_classes = checkpoint.get("num_classes", config.NUM_CLASSES)
            global_mean = checkpoint.get("global_mean", None)
            global_std = checkpoint.get("global_std", None)
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint

        if state_dict is not None and "input_proj.0.weight" in state_dict:
            input_dim = state_dict["input_proj.0.weight"].shape[1]
            d_model = state_dict["input_proj.0.weight"].shape[0]
            print(f"[+] Detected weights configuration: input_dim={input_dim}, d_model={d_model}")
        elif state_dict is not None and "input_proj.weight" in state_dict:
            input_dim = state_dict["input_proj.weight"].shape[1]
            d_model = state_dict["input_proj.weight"].shape[0]
            print(f"[+] Detected weights configuration: input_dim={input_dim}, d_model={d_model}")
    else:
        print(f"[!] Warning: Model weights '{target_path}' not found. Initializing architecture.")

    model = BinaryEngagementTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=config.DROPOUT,
        num_classes=num_classes
    )

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
            print("    Successfully loaded model state_dict.")
        except Exception as e:
            print(f"[!] Warning: Could not strictly load state_dict ({e}), attempting non-strict load...")
            model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    if global_mean is not None and isinstance(global_mean, torch.Tensor):
        global_mean = global_mean.to(device)
    if global_std is not None and isinstance(global_std, torch.Tensor):
        global_std = global_std.to(device)

    return model, input_dim, global_mean, global_std


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Test Classroom Video with Binary Engagement Transformer")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to classroom video file (auto-detects from 'clasroom_recording' if omitted)")
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Folder containing classroom recordings")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained binary_transformer_weights.pth")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for annotated video and reports")
    parser.add_argument("--yolo_model", type=str, default="yolov8n-face.pt",
                        help="YOLO model path (yolov8n-face.pt or yolo11n.pt)")
    parser.add_argument("--seq_len", type=int, default=120,
                        help="Sliding buffer sequence length for student temporal prediction (default: 120)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="YOLO detection confidence threshold (default: 0.35)")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Engagement decision threshold (default: 0.50 | Lower to 0.40-0.45 for more Engaged, raise to 0.55-0.65 for stricter Disengaged)")
    parser.add_argument("--min_frames", type=int, default=12,
                        help="Minimum frames to confirm a legitimate student track (default: 12)")
    parser.add_argument("--save_frames", action="store_true", default=True,
                        help="Save extracted annotated frames to Google Drive (default: True)")
    parser.add_argument("--no_save_frames", dest="save_frames", action="store_false",
                        help="Do not save extracted individual frames")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("  BINARY ENGAGEMENT TRANSFORMER — CLASSROOM VIDEO TESTING")
    print("  Classes : 0 = Engaged (GREEN)  |  1 = Disengaged (RED)")
    print(f"  Device  : {device}")
    print("=" * 65)

    if YOLO is None:
        print("[Error] 'ultralytics' is required. Install with: pip install ultralytics")
        sys.exit(1)

    # 1. Resolve video path
    video_path = find_classroom_video(args.video, args.video_dir)
    if not video_path or not os.path.exists(video_path):
        print("\n[Error] No classroom video could be found.")
        print("Please provide the path using --video:")
        print("  python test_classroom_binary_transformer.py --video /path/to/clasroom_recording/video.mp4")
        sys.exit(1)

    print(f"\n[+] Input Video: {video_path}")

    # 2. Load model
    transformer_model, input_dim, global_mean, global_std = load_binary_transformer(args.model, device)

    # 3. Load YOLO Detector & Tracker
    yolo_weight = args.yolo_model
    # Check if face weight exists or fallback
    if "face" in yolo_weight and not os.path.exists(yolo_weight):
        # Auto-download face model from known mirrors if possible, or fallback
        face_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
        try:
            print(f"[+] Attempting to download {yolo_weight}...")
            import urllib.request
            urllib.request.urlretrieve(face_url, yolo_weight)
            print("    Download complete.")
        except Exception:
            print(f"[!] Could not download '{yolo_weight}'. Falling back to standard 'yolo11n.pt' / 'yolov8n.pt'.")
            yolo_weight = "yolo11n.pt"

    print(f"[+] Loading YOLO model: {yolo_weight}")
    yolo = YOLO(yolo_weight)
    is_face_model = "face" in yolo_weight.lower()

    # 4. Open video source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Failed to open video file: {video_path}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output directory on Google Drive
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = args.out_dir if args.out_dir else os.path.join(
        config.EVAL_OUT_DIR, f"{video_stem}_binary_results"
    )

    # If running on Colab and output path is in ephemeral local storage, redirect to Google Drive
    if config.RUNNING_ON_COLAB and out_dir.startswith("/content/") and not out_dir.startswith("/content/drive/"):
        print(f"[!] Warning: Specified --out_dir '{out_dir}' is in ephemeral local storage.")
        print(f"[+] Redirecting to Google Drive to ensure persistent storage.")
        out_dir = os.path.join(config.EVAL_OUT_DIR, f"{video_stem}_binary_results")

    os.makedirs(out_dir, exist_ok=True)

    # Extracted frames directory on Google Drive
    frames_save_dir = os.path.join(out_dir, "extracted_frames")
    if args.save_frames:
        os.makedirs(frames_save_dir, exist_ok=True)

    output_video_path = os.path.join(out_dir, f"{video_stem}_annotated_binary.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 5. Initialize visual & OpenFace dynamics feature extractor (1280 CNN + 709 OpenFace = 1989)
    feature_pipeline = FusedFeaturePipeline(
        device=device,
        global_mean=global_mean,
        global_std=global_std
    )
    last_of_feats = {}

    # Initialize Face Cascade for human face verification
    face_cascade = None
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        pass

    # Class mappings & Colors (BGR format for OpenCV)
    # 0 = Engaged (GREEN: 0, 255, 0)
    # 1 = Disengaged (RED: 0, 0, 255)
    class_names = {
        0: "Engaged",
        1: "Disengaged"
    }
    colors = {
        0: (0, 255, 0),    # Green for Engaged
        1: (0, 0, 255)     # Red for Disengaged
    }

    # Tracking & metrics data structures
    face_buffers = {}            # track_id -> list of feature vectors
    predictions_history = {}     # track_id -> dict of frame_idx -> pred_class
    confirmed_face_tracks = set()# track_ids with verified human face
    detailed_records = []        # per-frame record for CSV
    frame_metrics = []           # classroom engagement per frame

    print(f"[+] Video Resolution : {width}x{height} @ {fps:.1f} FPS ({total_frames} total frames)")
    print(f"[+] Output Directory : {out_dir}")
    print(f"[+] Annotated Video  : {output_video_path}")
    print(f"[+] Confidence Thresh: {args.conf}")
    print(f"[+] Decision Thresh  : {args.threshold} (Higher = Stricter)")
    print(f"[+] Min Track Frames : {args.min_frames}")
    print("\n[+] Processing Video Frames & Generating Annotations...")

    processed_frames = 0
    start_time = time.time()

    for frame_idx in tqdm(range(total_frames), desc="Annotating Classroom Video"):
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1
        timestamp_sec = round(processed_frames / fps, 2)

        # Run YOLO with ByteTrack (strictly person class [0] if generic object detector)
        track_kwargs = {
            "source": frame,
            "persist": True,
            "conf": args.conf,
            "tracker": "bytetrack.yaml",
            "verbose": False
        }
        if not is_face_model:
            track_kwargs["classes"] = [0]  # PERSON ONLY: Strictly rejects bags, chairs, laptops, tables

        try:
            results = yolo.track(**track_kwargs)
        except Exception:
            track_kwargs["tracker"] = "botsort.yaml"
            results = yolo.track(**track_kwargs)

        current_frame_engaged = 0
        current_frame_disengaged = 0
        current_frame_students = 0

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls = int(box.cls[0].item()) if box.cls is not None else 0
                if not is_face_model and cls != 0:
                    continue  # Strictly reject any non-person (chair, backpack, etc.)

                track_id = int(box.id[0].item()) if box.id is not None else 0
                if track_id <= 0:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = map(int, xyxy)
                bw, bh = xmax - xmin, ymax - ymin

                # Minimum size and aspect ratio filter
                if bw < 20 or bh < 25:
                    continue
                if not is_face_model:
                    aspect = bw / float(bh)
                    if aspect > 1.6 or aspect < 0.22:
                        continue

                xmin_c, ymin_c = max(0, xmin), max(0, ymin)
                xmax_c, ymax_c = min(width, xmax), min(height, ymax)

                # 2. Extract Head/Face Region (Keep track of faces properly)
                face_crop = None
                if not is_face_model:
                    # Upper 35% of person bounding box corresponds to head/face area
                    head_h = max(20, int(bh * 0.35))
                    head_ymax = min(height, ymin_c + head_h)
                    head_crop = frame[ymin_c:head_ymax, xmin_c:xmax_c]

                    if face_cascade is not None and head_crop.size > 0:
                        head_gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            head_gray, scaleFactor=1.1, minNeighbors=3, minSize=(16, 16)
                        )
                        if len(faces) > 0:
                            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                            pad_x, pad_y = int(fw * 0.10), int(fh * 0.10)
                            fx1 = max(0, xmin_c + fx - pad_x)
                            fy1 = max(0, ymin_c + fy - pad_y)
                            fx2 = min(width, xmin_c + fx + fw + pad_x)
                            fy2 = min(height, ymin_c + fy + fh + pad_y)
                            face_crop = frame[fy1:fy2, fx1:fx2]
                    if face_crop is None or face_crop.size == 0:
                        face_crop = head_crop
                else:
                    face_crop = frame[ymin_c:ymax_c, xmin_c:xmax_c]

                if face_crop is None or face_crop.shape[0] < 12 or face_crop.shape[1] < 12:
                    continue

                confirmed_face_tracks.add(track_id)

                # Extract 1989-dim fused features (1280 CNN + 709 OpenFace dynamics)
                prev_of = last_of_feats.get(track_id, None)
                fused_tensor, new_of = feature_pipeline.extract_and_fuse(
                    face_crop, previous_of_feat=prev_of
                )
                last_of_feats[track_id] = new_of

                # Update student temporal sliding buffer
                if track_id not in face_buffers:
                    face_buffers[track_id] = []
                    predictions_history[track_id] = {}

                face_buffers[track_id].append(fused_tensor.view(-1))
                if len(face_buffers[track_id]) > args.seq_len:
                    face_buffers[track_id].pop(0)

                # Form sequence for transformer with masked zero padding (exact match to training)
                actual_len = len(face_buffers[track_id])
                seq_tensors = torch.stack([s.view(-1) for s in face_buffers[track_id]]).to(device)
                if actual_len < args.seq_len:
                    pad_tensor = torch.zeros((args.seq_len - actual_len, 1989), dtype=torch.float32, device=device)
                    seq_tensor = torch.cat([seq_tensors, pad_tensor], dim=0).unsqueeze(0)
                else:
                    seq_tensor = seq_tensors[-args.seq_len:].unsqueeze(0)
                    actual_len = args.seq_len

                # Model Prediction: Direct output from trained Binary Transformer
                with torch.no_grad():
                    logits = transformer_model(seq_tensor, seq_lens=[actual_len])
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                eng_prob = float(probs[0])
                diseng_prob = float(probs[1])

                # Engagement threshold decision rule
                if eng_prob >= args.threshold:
                    pred_class = 0
                    confidence = eng_prob
                else:
                    pred_class = 1
                    confidence = diseng_prob

                predictions_history[track_id][processed_frames] = pred_class

                # Only draw annotation if track persisted for at least 3 frames
                if len(face_buffers[track_id]) >= 3:
                    current_frame_students += 1
                    if pred_class == 0:
                        current_frame_engaged += 1
                    else:
                        current_frame_disengaged += 1

                    box_color = colors[pred_class]
                    box_label = f"ID {track_id}: {class_names[pred_class]} ({confidence * 100:.0f}%)"

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(box_label, font, font_scale, thickness)
                    badge_ymin = max(ymin - text_h - 10, 0)
                    badge_ymax = ymin

                    cv2.rectangle(
                        frame,
                        (xmin, badge_ymin),
                        (xmin + text_w + 10, badge_ymax),
                        box_color,
                        -1
                    )
                    cv2.putText(
                        frame,
                        box_label,
                        (xmin + 5, badge_ymax - 4),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA
                    )

                detailed_records.append({
                    "frame": processed_frames,
                    "timestamp_sec": timestamp_sec,
                    "student_id": track_id,
                    "predicted_class": pred_class,
                    "engagement_label": class_names[pred_class],
                    "confidence_pct": round(confidence * 100, 2),
                    "engaged_prob": round(float(probs[0]), 4),
                    "disengaged_prob": round(float(probs[1]), 4),
                    "box_xmin": xmin,
                    "box_ymin": ymin,
                    "box_xmax": xmax,
                    "box_ymax": ymax
                })

        # ── Top HUD Classroom Analytics Overlay ─────────────────────────────
        hud_h = 44
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (width, hud_h), (20, 24, 30), -1)
        cv2.addWeighted(hud_overlay, 0.82, frame, 0.18, 0, frame)

        time_str = f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}"
        cv2.putText(
            frame,
            f"Classroom Binary Analysis | Time: {time_str} (F:{processed_frames})",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        eng_rate = (current_frame_engaged / current_frame_students * 100) if current_frame_students > 0 else 0
        frame_metrics.append({
            "frame": processed_frames,
            "timestamp_sec": timestamp_sec,
            "students": current_frame_students,
            "engaged": current_frame_engaged,
            "disengaged": current_frame_disengaged,
            "engagement_rate": eng_rate
        })

        # Right-aligned HUD stats
        stats_text = (
            f"Students: {current_frame_students}  |  "
            f"Engaged: {current_frame_engaged}  |  "
            f"Disengaged: {current_frame_disengaged}  |  "
            f"Rate: {eng_rate:.0f}%"
        )
        (tw, _), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(
            frame,
            stats_text,
            (max(width - tw - 20, 350), 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0) if eng_rate >= 50 else (0, 140, 255),
            1,
            cv2.LINE_AA
        )

        out_video.write(frame)
        if args.save_frames:
            frame_filename = f"frame_{processed_frames:05d}.jpg"
            cv2.imwrite(
                os.path.join(frames_save_dir, frame_filename),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )

    cap.release()
    out_video.release()

    elapsed = time.time() - start_time
    avg_fps = processed_frames / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 65)
    print("  PROCESSING COMPLETE")
    print("=" * 65)
    print(f"  Processed Frames  : {processed_frames} / {total_frames}")
    print(f"  Processing Speed  : {avg_fps:.2f} FPS")
    print(f"  Annotated Video   : {output_video_path}")
    if args.save_frames:
        print(f"  Extracted Frames  : {frames_save_dir} (Google Drive)")

    # Generate analytical reports and charts
    export_summary_reports(
        predictions_history,
        detailed_records,
        frame_metrics,
        class_names,
        out_dir,
        video_stem,
        min_frames=args.min_frames,
        confirmed_face_tracks=confirmed_face_tracks,
        is_face_model=is_face_model,
        total_frames=processed_frames
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORT CSV, TEXT REPORT, AND ANALYTICS CHART
# ─────────────────────────────────────────────────────────────────────────────
def export_summary_reports(predictions_history, detailed_records, frame_metrics,
                           class_names, out_dir, video_stem, min_frames=12,
                           confirmed_face_tracks=None, is_face_model=False,
                           total_frames=100):
    """
    Saves:
      1. Detailed per-frame predictions CSV.
      2. Comprehensive Student Engagement Text Report.
      3. Visual summary analytics chart (Timeline & Per-Student Engagement).
    """
    # 1. Save detailed CSV
    csv_path = os.path.join(out_dir, f"{video_stem}_binary_details.csv")
    if detailed_records:
        df = pd.DataFrame(detailed_records)
        df.to_csv(csv_path, index=False)
        print(f"  Detailed CSV Log  : {csv_path}")

    # 2. Text Summary Report
    txt_path = os.path.join(out_dir, f"{video_stem}_binary_report.txt")
    report_lines = [
        "=================================================================",
        "  BINARY ENGAGEMENT TRANSFORMER — CLASSROOM EVALUATION REPORT",
        f"  Video: {video_stem}",
        "=================================================================",
        ""
    ]

    effective_min = max(min_frames, int(total_frames * 0.04))
    student_summary = []
    total_engaged_frames = 0
    total_disengaged_frames = 0

    for track_id, frames_dict in sorted(predictions_history.items()):
        total_tracked = len(frames_dict)
        if total_tracked < effective_min:  # Skip transient detections
            continue
        if confirmed_face_tracks is not None and not is_face_model:
            if track_id not in confirmed_face_tracks and total_tracked < int(total_frames * 0.20):
                continue

        c_engaged = sum(1 for p in frames_dict.values() if p == 0)
        c_disengaged = sum(1 for p in frames_dict.values() if p == 1)

        pct_engaged = (c_engaged / total_tracked) * 100
        pct_disengaged = (c_disengaged / total_tracked) * 100

        total_engaged_frames += c_engaged
        total_disengaged_frames += c_disengaged

        status = "ENGAGED" if pct_engaged >= 50.0 else "DISENGAGED"

        report_lines.append(f"Student #{track_id} — Overall: {status}")
        report_lines.append(f"  Tracked Frames : {total_tracked}")
        report_lines.append(f"  Engaged        : {pct_engaged:5.1f}% ({c_engaged} frames) [GREEN]")
        report_lines.append(f"  Disengaged     : {pct_disengaged:5.1f}% ({c_disengaged} frames) [RED]")
        report_lines.append("-" * 45)

        student_summary.append({
            "student": f"ID {track_id}",
            "engaged_pct": pct_engaged,
            "disengaged_pct": pct_disengaged,
            "frames": total_tracked,
            "status": status
        })

    total_detections = total_engaged_frames + total_disengaged_frames
    overall_eng_pct = (total_engaged_frames / total_detections * 100) if total_detections > 0 else 0

    report_lines.append("")
    report_lines.append("CLASSROOM OVERALL SUMMARY:")
    report_lines.append(f"  Total Tracked Students : {len(student_summary)}")
    report_lines.append(f"  Overall Engagement Rate: {overall_eng_pct:.1f}%")
    report_lines.append("=================================================================")

    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Text Report       : {txt_path}")

    # 3. Analytics Chart (Per-Student Engagement + Timeline)
    if student_summary:
        chart_path = os.path.join(out_dir, f"{video_stem}_binary_chart.png")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Subplot 1: Per-student stacked bar chart
        students = [s["student"] for s in student_summary]
        eng_vals = [s["engaged_pct"] for s in student_summary]
        dis_vals = [s["disengaged_pct"] for s in student_summary]
        x = np.arange(len(students))

        ax1.bar(x, eng_vals, label="Engaged (Green)", color="#2ecc71", width=0.55)
        ax1.bar(x, dis_vals, bottom=eng_vals, label="Disengaged (Red)", color="#e74c3c", width=0.55)
        ax1.set_ylabel("Percentage (%)", fontsize=11)
        ax1.set_title(f"Per-Student Engagement Distribution — {video_stem}", fontsize=12, pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(students, rotation=35, ha="right")
        ax1.set_ylim(0, 100)
        ax1.legend(loc="upper right")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # Subplot 2: Classroom Engagement Timeline
        if frame_metrics:
            df_m = pd.DataFrame(frame_metrics)
            # Smooth with rolling average
            rolling_rate = df_m["engagement_rate"].rolling(window=15, min_periods=1).mean()
            ax2.plot(df_m["timestamp_sec"], rolling_rate, color="#3498db", lw=2, label="Engagement Rate (%)")
            ax2.axhline(50, color="gray", linestyle=":", label="50% Threshold")
            ax2.set_xlabel("Video Timestamp (seconds)", fontsize=11)
            ax2.set_ylabel("Engagement Rate (%)", fontsize=11)
            ax2.set_title("Classroom Overall Engagement Over Time", fontsize=12, pad=10)
            ax2.set_ylim(0, 105)
            ax2.legend(loc="lower right")
            ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=250)
        plt.close()
        print(f"  Analytics Chart   : {chart_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
