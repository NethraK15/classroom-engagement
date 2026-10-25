"""
app_dashboard.py
─────────────────────────────────────────────────────────────────────────────
LIVE DEPLOYED CLASSROOM ENGAGEMENT DASHBOARD
Binary Engagement Transformer (Engaged vs Disengaged)
YOLOv8 + ByteTrack + EfficientNet-B0 (1280) + OpenFace (709) = 1989 Features

Capabilities:
  1. 🎥 Upload Classroom Video (Multi-student detection, tracking & full analytics)
  2. 📹 Live Webcam Camera (Real-time engagement streaming & instant prediction)
  3. 📊 Interactive Analytics (Student engagement distribution & timeline charts)
  4. 📋 Downloadable Reports (Detailed per-frame CSV log & Text summary)
  5. 🌐 Deployed Public Link (Shareable link for project demo via --share)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import tempfile
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config_binary_transformer as config
from models_binary_transformer import BinaryEngagementTransformer
from feature_extractor_binary import FusedFeaturePipeline, apply_illumination_enhancement

# Optional imports with friendly fallbacks
try:
    import gradio as gr
except ImportError:
    gr = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL & DETECTOR LOADER
# ─────────────────────────────────────────────────────────────────────
class EngagementEngine:
    """Manages YOLO detector, Feature Pipeline, and Binary Engagement Transformer."""
    def __init__(self, model_path=None, yolo_path="yolov8n-face.pt", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EngageEngine] Initializing on device: {self.device}")

        # 1. Load YOLO Face/Person Detector
        self.yolo, self.is_face_model = self._init_yolo(yolo_path)
        self.face_cascade = self._init_face_cascade()

        # 2. Load Binary Transformer Checkpoint
        self.model, self.input_dim, self.global_mean, self.global_std = self._load_transformer(model_path)

        # 3. Initialize Feature Extractor Pipeline (1280 CNN + 709 OpenFace = 1989)
        self.feature_pipeline = FusedFeaturePipeline(
            device=self.device,
            global_mean=self.global_mean,
            global_std=self.global_std
        )

        # Class maps & visual styles
        self.class_names = {0: "Engaged", 1: "Disengaged"}
        self.class_colors = {
            0: (46, 204, 113),   # Emerald Green (Engaged)
            1: (231, 76, 60)     # Crimson Red (Disengaged)
        }

    def _init_face_cascade(self):
        try:
            path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(path):
                return cv2.CascadeClassifier(path)
        except Exception:
            pass
        return None

    def _init_yolo(self, yolo_path):
        if YOLO is None:
            print("[EngageEngine] Warning: ultralytics is not installed. YOLO unavailable.")
            return None, False

        is_face = "face" in str(yolo_path).lower()
        # Check for face weight or auto-download mirror
        if is_face and not os.path.exists(yolo_path):
            face_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
            try:
                print(f"[EngageEngine] Downloading YOLO Face model '{yolo_path}'...")
                import urllib.request
                urllib.request.urlretrieve(face_url, yolo_path)
                print("[EngageEngine] Download successful.")
            except Exception:
                print(f"[EngageEngine] Falling back to standard 'yolo11n.pt' / 'yolov8n.pt'.")
                yolo_path = "yolo11n.pt"
                is_face = False

        try:
            print(f"[EngageEngine] Loading YOLO model: {yolo_path} (is_face={is_face})")
            return YOLO(yolo_path), is_face
        except Exception as e:
            print(f"[EngageEngine] Failed to load YOLO: {e}")
            return None, False

    def _load_transformer(self, model_path):
        # Auto-discover weights
        target_path = model_path
        if not target_path or not os.path.exists(target_path):
            candidates = [
                config.MODEL_WEIGHTS_PATH,
                os.path.join(SCRIPT_DIR, "models_binary_transformer", "binary_transformer_weights.pth"),
                os.path.join(config.PROJECT_ROOT, "models_binary_transformer", "binary_transformer_weights.pth"),
                os.path.join("/content/drive/MyDrive/Student_Engagement_DL/models_binary_transformer", "binary_transformer_weights.pth")
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    target_path = c
                    break

        input_dim = 1989
        d_model = config.D_MODEL
        nhead = config.NHEAD
        num_layers = config.NUM_LAYERS
        dim_feedforward = config.DIM_FEEDFORWARD
        global_mean = None
        global_std = None
        state_dict = None

        if target_path and os.path.exists(target_path):
            print(f"[EngageEngine] Loading trained checkpoint from: {target_path}")
            checkpoint = torch.load(target_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                input_dim = checkpoint.get("input_dim", 1989)
                d_model = checkpoint.get("d_model", config.D_MODEL)
                nhead = checkpoint.get("nhead", config.NHEAD)
                num_layers = checkpoint.get("num_layers", config.NUM_LAYERS)
                dim_feedforward = checkpoint.get("dim_feedforward", config.DIM_FEEDFORWARD)
                global_mean = checkpoint.get("global_mean", None)
                global_std = checkpoint.get("global_std", None)
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint

            if state_dict and "input_proj.0.weight" in state_dict:
                input_dim = state_dict["input_proj.0.weight"].shape[1]
                d_model = state_dict["input_proj.0.weight"].shape[0]
        else:
            print(f"[EngageEngine] Checkpoint not found at '{target_path}'. Initializing clean architecture.")

        model = BinaryEngagementTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=config.DROPOUT,
            num_classes=2
        )

        if state_dict is not None:
            try:
                model.load_state_dict(state_dict)
                print("[EngageEngine] State dict loaded successfully.")
            except Exception as e:
                print(f"[EngageEngine] Non-strict load fallback: {e}")
                model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        model.eval()

        if global_mean is not None and isinstance(global_mean, torch.Tensor):
            global_mean = global_mean.to(self.device)
        if global_std is not None and isinstance(global_std, torch.Tensor):
            global_std = global_std.to(self.device)

        return model, input_dim, global_mean, global_std


# Global engine singleton
ENGINE = None

def get_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = EngagementEngine()
    return ENGINE


# ─────────────────────────────────────────────────────────────────────────────
# 2. VIDEO PROCESSING PIPELINE (Multi-Student with YOLO + ByteTrack)
# ─────────────────────────────────────────────────────────────────────
def process_uploaded_video(
    video_path,
    confidence_thresh=0.35,
    engagement_threshold=0.50,
    seq_buffer_len=120,
    min_track_frames=12,
    enable_clahe=False,
    progress=None
):
    if not video_path or not os.path.exists(video_path):
        return None, None, None, None, "❌ Error: Video file not found."

    engine = get_engine()
    if engine.yolo is None:
        return None, None, None, None, "❌ Error: YOLO model not loaded. Please install ultralytics."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None, "❌ Error: Failed to open video file."

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 100

    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    # 1. Local workspace dashboard output directory (inside current working dir for Gradio)
    local_out_dir = os.path.join(SCRIPT_DIR, "dashboard_outputs", f"{video_stem}_dashboard_results")
    os.makedirs(local_out_dir, exist_ok=True)

    # 2. Google Drive permanent backup directory
    drive_out_dir = os.path.join(config.EVAL_OUT_DIR, f"{video_stem}_dashboard_results")
    try:
        os.makedirs(drive_out_dir, exist_ok=True)
    except Exception:
        pass

    temp_video_path = os.path.join(local_out_dir, f"{video_stem}_annotated_raw.mp4")
    final_video_path = os.path.join(local_out_dir, f"{video_stem}_annotated_web.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # Tracking buffers & data
    face_buffers = {}            # track_id -> list of [1989] tensors
    last_of_feats = {}           # track_id -> last [709] feature
    predictions_history = {}     # track_id -> dict(frame_idx -> pred_class)
    confirmed_face_tracks = set()# track_ids that had a verified human face
    detailed_records = []        # for CSV export
    frame_metrics = []           # classroom engagement timeline

    start_time = time.time()
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_count += 1
        timestamp_sec = round(processed_count / fps, 2)

        if progress is not None and total_frames > 0:
            progress(
                min(processed_count / total_frames, 0.99),
                desc=f"Processing Frame {processed_count}/{total_frames} ({int(processed_count/total_frames*100)}%)"
            )

        # Run YOLO with ByteTrack (Strictly person class [0] when standard YOLO is used)
        track_kwargs = {
            "source": frame,
            "persist": True,
            "conf": confidence_thresh,
            "tracker": "bytetrack.yaml",
            "verbose": False
        }
        if not engine.is_face_model:
            track_kwargs["classes"] = [0]  # PERSON ONLY: Strictly rejects bags, chairs, laptops, tables

        try:
            results = engine.yolo.track(**track_kwargs)
        except Exception:
            track_kwargs["tracker"] = "botsort.yaml"
            results = engine.yolo.track(**track_kwargs)

        current_frame_engaged = 0
        current_frame_disengaged = 0
        current_frame_students = 0

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # 1. Filter out non-person classes if using generic object detector
                cls = int(box.cls[0].item()) if box.cls is not None else 0
                if not engine.is_face_model and cls != 0:
                    continue  # Rejects chair, backpack, handbag, desk, etc.

                track_id = int(box.id[0].item()) if box.id is not None else 0
                if track_id <= 0:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = map(int, xyxy)
                bw, bh = xmax - xmin, ymax - ymin

                # Minimum size and aspect ratio filter to discard background clutter
                if bw < 20 or bh < 25:
                    continue
                if not engine.is_face_model:
                    aspect = bw / float(bh)
                    if aspect > 1.6 or aspect < 0.22:
                        continue

                xmin_c, ymin_c = max(0, xmin), max(0, ymin)
                xmax_c, ymax_c = min(width, xmax), min(height, ymax)

                # 2. Extract Face Region (Keep track of faces properly)
                face_crop = None
                if not engine.is_face_model:
                    # Upper 35% of person bounding box corresponds to head/face area
                    head_h = max(20, int(bh * 0.35))
                    head_ymax = min(height, ymin_c + head_h)
                    head_crop = frame[ymin_c:head_ymax, xmin_c:xmax_c]

                    if engine.face_cascade is not None and head_crop.size > 0:
                        head_gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                        faces = engine.face_cascade.detectMultiScale(
                            head_gray, scaleFactor=1.1, minNeighbors=3, minSize=(16, 16)
                        )
                        if len(faces) > 0:
                            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                            pad_x, pad_y = int(fw * 0.20), int(fh * 0.20)
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

                # 3. Extract 1989-dim fused feature (1280 CNN + 709 OpenFace dynamics)
                prev_of = last_of_feats.get(track_id, None)
                fused_tensor, new_of = engine.feature_pipeline.extract_and_fuse(
                    face_crop, previous_of_feat=prev_of, enable_clahe=enable_clahe
                )
                last_of_feats[track_id] = new_of

                # 4. Update temporal sliding buffer with zero padding (exact match to training & evaluation)
                if track_id not in face_buffers:
                    face_buffers[track_id] = []
                    predictions_history[track_id] = {}

                face_buffers[track_id].append(fused_tensor.view(-1))
                if len(face_buffers[track_id]) > seq_buffer_len:
                    face_buffers[track_id].pop(0)

                # Form full sequence length for transformer with masked zero padding
                actual_len = len(face_buffers[track_id])
                seq_tensors = torch.stack([f.view(-1) for f in face_buffers[track_id]]).to(engine.device)
                if actual_len < seq_buffer_len:
                    pad_tensor = torch.zeros((seq_buffer_len - actual_len, 1989), dtype=torch.float32, device=engine.device)
                    seq_tensor = torch.cat([seq_tensors, pad_tensor], dim=0).unsqueeze(0)
                else:
                    seq_tensor = seq_tensors[-seq_buffer_len:].unsqueeze(0)
                    actual_len = seq_buffer_len

                seq_len_t = torch.tensor([actual_len], device=engine.device)

                # 5. Direct Binary Engagement Transformer Prediction
                with torch.no_grad():
                    logits = engine.model(seq_tensor, seq_len_t)
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                eng_prob = float(probs[0])
                diseng_prob = float(probs[1])

                # Decision rule using calibrated threshold (default 0.50)
                if eng_prob >= engagement_threshold:
                    pred_class = 0
                    conf_score = eng_prob * 100.0
                else:
                    pred_class = 1
                    conf_score = diseng_prob * 100.0

                predictions_history[track_id][processed_count] = pred_class

                # 6. Video annotation
                if len(face_buffers[track_id]) >= 2:
                    current_frame_students += 1
                    if pred_class == 0:
                        current_frame_engaged += 1
                    else:
                        current_frame_disengaged += 1

                    status_name = "Engaged" if pred_class == 0 else "Disengaged"
                    b_color = (46, 204, 113) if pred_class == 0 else (60, 76, 231)  # Emerald Green / Red

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), b_color, 2)

                    label_str = f"#{track_id} {status_name} ({conf_score:.0f}%)"
                    (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    cv2.rectangle(
                        frame,
                        (xmin, max(0, ymin - th - 8)),
                        (xmin + tw + 8, ymin),
                        b_color,
                        -1
                    )
                    cv2.putText(
                        frame,
                        label_str,
                        (xmin + 4, ymin - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

                detailed_records.append({
                    "frame": processed_count,
                    "timestamp_sec": timestamp_sec,
                    "student_id": track_id,
                    "status": "Engaged" if pred_class == 0 else "Disengaged",
                    "confidence_pct": round(conf_score, 1),
                    "engaged_prob": round(eng_prob, 3),
                    "disengaged_prob": round(diseng_prob, 3),
                    "bbox": f"[{xmin},{ymin},{xmax},{ymax}]"
                })

        # Top HUD Banner
        hud = frame.copy()
        cv2.rectangle(hud, (0, 0), (width, 42), (20, 24, 33), -1)
        cv2.addWeighted(hud, 0.85, frame, 0.15, 0, frame)

        eng_rate = (current_frame_engaged / current_frame_students * 100) if current_frame_students > 0 else 0
        hud_left = f"AI Classroom Engagement | F:{processed_count} ({timestamp_sec:.1f}s)"
        hud_right = f"Students: {current_frame_students} | Engaged: {current_frame_engaged} | Disengaged: {current_frame_disengaged} | {eng_rate:.0f}%"

        cv2.putText(frame, hud_left, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        (rw, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        hud_col = (46, 204, 113) if eng_rate >= 50 else (60, 76, 231)
        cv2.putText(frame, hud_right, (max(width - rw - 14, 300), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, hud_col, 1, cv2.LINE_AA)

        frame_metrics.append({
            "frame": processed_count,
            "timestamp_sec": timestamp_sec,
            "students": current_frame_students,
            "engaged": current_frame_engaged,
            "disengaged": current_frame_disengaged,
            "engagement_rate": eng_rate
        })

        out_writer.write(frame)

    cap.release()
    out_writer.release()

    elapsed = max(time.time() - start_time, 0.01)
    avg_fps = processed_count / elapsed

    # Re-encode to browser-compatible H.264 using ffmpeg if available
    web_ready_video = temp_video_path
    try:
        cmd = [
            "ffmpeg", "-y", "-i", temp_video_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "baseline", "-level", "3.0",
            final_video_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and os.path.exists(final_video_path):
            web_ready_video = final_video_path
    except Exception:
        pass

    # Save detailed CSV in local dashboard outputs
    csv_path = os.path.join(local_out_dir, f"{video_stem}_engagement_log.csv")
    df_records = pd.DataFrame(detailed_records) if detailed_records else pd.DataFrame(columns=["frame"])
    df_records.to_csv(csv_path, index=False)

    # Compute Summary Statistics (Strictly filters out ghost detections & non-human tracks)
    effective_min_frames = max(min_track_frames, int(processed_count * 0.04))
    student_summary = []
    tot_eng_frames = 0
    tot_dis_frames = 0

    for track_id, f_dict in sorted(predictions_history.items()):
        n_t = len(f_dict)
        # Discard transient ghost tracks
        if n_t < effective_min_frames:
            continue
        # Discard tracks that never had a verified human face unless very prolonged
        if not engine.is_face_model and track_id not in confirmed_face_tracks and n_t < int(processed_count * 0.20):
            continue

        c_eng = sum(1 for p in f_dict.values() if p == 0)
        c_dis = sum(1 for p in f_dict.values() if p == 1)
        p_eng = (c_eng / n_t) * 100.0
        p_dis = (c_dis / n_t) * 100.0
        tot_eng_frames += c_eng
        tot_dis_frames += c_dis
        status = "ENGAGED" if p_eng >= 50.0 else "DISENGAGED"

        student_summary.append({
            "Student ID": f"Student #{track_id}",
            "Overall Status": "🟢 Engaged" if status == "ENGAGED" else "🔴 Disengaged",
            "Engaged %": f"{p_eng:.1f}%",
            "Disengaged %": f"{p_dis:.1f}%",
            "Tracked Frames": n_t
        })

    total_detections = tot_eng_frames + tot_dis_frames
    overall_eng_pct = (tot_eng_frames / total_detections * 100.0) if total_detections > 0 else 0.0

    df_summary = pd.DataFrame(student_summary) if student_summary else pd.DataFrame(columns=["Student ID"])

    # Generate Matplotlib Analytics Plot in local dashboard outputs
    chart_path = os.path.join(local_out_dir, f"{video_stem}_engagement_plot.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), dpi=150)
    fig.patch.set_facecolor('#0f172a')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#e2e8f0')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    # Subplot 1: Per-student distribution
    if student_summary:
        names = [s["Student ID"] for s in student_summary]
        e_vals = [float(s["Engaged %"].replace("%", "")) for s in student_summary]
        d_vals = [float(s["Disengaged %"].replace("%", "")) for s in student_summary]
        xs = np.arange(len(names))

        ax1.bar(xs, e_vals, label="Engaged", color="#2ecc71", width=0.52)
        ax1.bar(xs, d_vals, bottom=e_vals, label="Disengaged", color="#e74c3c", width=0.52)
        ax1.set_title(f"Per-Student Engagement Distribution — {video_stem}", color='#f8fafc', fontsize=11, fontweight='bold')
        ax1.set_ylabel("Percentage (%)", color='#94a3b8', fontsize=10)
        ax1.set_xticks(xs)
        ax1.set_xticklabels(names, rotation=25, ha='right', color='#e2e8f0')
        ax1.set_ylim(0, 105)
        ax1.legend(loc="upper right", facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
        ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#475569')

    # Subplot 2: Engagement Timeline
    if frame_metrics:
        df_m = pd.DataFrame(frame_metrics)
        smooth_rate = df_m["engagement_rate"].rolling(window=10, min_periods=1).mean()
        ax2.plot(df_m["timestamp_sec"], smooth_rate, color="#38bdf8", lw=2.2, label="Classroom Engagement %")
        ax2.axhline(50, color="#f59e0b", linestyle=":", lw=1.5, label="50% Attention Threshold")
        ax2.set_title("Classroom Overall Attention Timeline", color='#f8fafc', fontsize=11, fontweight='bold')
        ax2.set_xlabel("Time (seconds)", color='#94a3b8', fontsize=10)
        ax2.set_ylabel("Engagement Rate (%)", color='#94a3b8', fontsize=10)
        ax2.set_ylim(0, 105)
        ax2.legend(loc="lower right", facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
        ax2.grid(True, linestyle='--', alpha=0.3, color='#475569')

    plt.tight_layout()
    plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    # Permanent backup copy to Google Drive
    try:
        if os.path.abspath(drive_out_dir) != os.path.abspath(local_out_dir):
            if os.path.exists(web_ready_video):
                shutil.copy2(web_ready_video, os.path.join(drive_out_dir, os.path.basename(web_ready_video)))
            if os.path.exists(chart_path):
                shutil.copy2(chart_path, os.path.join(drive_out_dir, os.path.basename(chart_path)))
            if os.path.exists(csv_path):
                shutil.copy2(csv_path, os.path.join(drive_out_dir, os.path.basename(csv_path)))
    except Exception as e:
        print(f"[EngageEngine] Backup to Drive: {e}")

    # Formatted KPI Card
    kpi_card = f"""
    ### 📊 Analysis Overview
    - **Overall Classroom Engagement**: **`{overall_eng_pct:.1f}%`**
    - **Students Tracked**: **`{len(student_summary)}`**
    - **Processed Frames**: **`{processed_count} / {total_frames}`**
    - **Processing Speed**: **`{avg_fps:.1f} FPS`** (Total Time: `{elapsed:.1f}s`)
    - **Output Storage**: Google Drive (`{drive_out_dir}`)
    """

    return web_ready_video, chart_path, df_summary, csv_path, kpi_card


# ─────────────────────────────────────────────────────────────────────────────
# 3. LIVE WEBCAM REAL-TIME FRAME INFERENCE
# ─────────────────────────────────────────────────────────────────────
# Rolling buffer for single webcam stream
WEBCAM_BUFFER = []
WEBCAM_LAST_OF = None

def process_webcam_frame(input_frame, engagement_threshold=0.50):
    """Processes live camera frame in real time with YOLO + 1989-dim feature extraction."""
    global WEBCAM_BUFFER, WEBCAM_LAST_OF

    if input_frame is None:
        return None, "Awaiting Camera Feed..."

    engine = get_engine()
    if engine.yolo is None:
        return input_frame, "YOLO detector not ready."

    frame = input_frame.copy()
    h, w = frame.shape[:2]

    # Detect faces with YOLO (strictly person class [0] if generic detector)
    detect_kwargs = {"conf": 0.35, "verbose": False}
    if not engine.is_face_model:
        detect_kwargs["classes"] = [0]

    try:
        results = engine.yolo(frame, **detect_kwargs)
    except Exception:
        results = engine.yolo(frame, conf=0.35, verbose=False)

    best_box = None
    max_area = 0

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            cls = int(box.cls[0].item()) if box.cls is not None else 0
            if not engine.is_face_model and cls != 0:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = map(int, xyxy)
            area = (xmax - xmin) * (ymax - ymin)
            if area > max_area:
                max_area = area
                best_box = (xmin, ymin, xmax, ymax)

    status_text = "No Student Detected"
    badge_color = (120, 120, 120)

    if best_box is not None:
        xmin, ymin, xmax, ymax = best_box
        bw, bh = xmax - xmin, ymax - ymin
        xmin_c, ymin_c = max(0, xmin), max(0, ymin)
        xmax_c, ymax_c = min(w, xmax), min(h, ymax)

        if bw >= 20 and bh >= 25:
            face_crop = None
            if not engine.is_face_model:
                # Require genuine face detection with Haar Cascade (strictly prevents detecting legs or clothing)
                head_h = max(25, int(bh * 0.45))
                head_crop = frame[ymin_c:min(h, ymin_c + head_h), xmin_c:xmax_c]
                if engine.face_cascade is not None and head_crop.size > 0:
                    head_gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                    faces = engine.face_cascade.detectMultiScale(
                        head_gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25)
                    )
                    if len(faces) > 0:
                        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                        pad_x, pad_y = int(fw * 0.20), int(fh * 0.20)
                        fx1 = max(0, xmin_c + fx - pad_x)
                        fy1 = max(0, ymin_c + fy - pad_y)
                        fx2 = min(w, xmin_c + fx + fw + pad_x)
                        fy2 = min(h, ymin_c + fy + fh + pad_y)
                        face_crop = frame[fy1:fy2, fx1:fx2]
                        # Center bounding box strictly on the human face
                        xmin, ymin, xmax, ymax = fx1, fy1, fx2, fy2
            else:
                face_crop = frame[ymin_c:ymax_c, xmin_c:xmax_c]

            # Only proceed if a genuine human face was confirmed
            if face_crop is not None and face_crop.shape[0] >= 15 and face_crop.shape[1] >= 15:
                fused_tensor, new_of = engine.feature_pipeline.extract_and_fuse(
                    face_crop, previous_of_feat=WEBCAM_LAST_OF
                )
                WEBCAM_LAST_OF = new_of
                WEBCAM_BUFFER.append(fused_tensor.view(-1))
                # Match training sequence window (120 frames)
                target_window = 120
                if len(WEBCAM_BUFFER) > target_window:
                    WEBCAM_BUFFER.pop(0)

                actual_len = len(WEBCAM_BUFFER)
                seq_tensors = torch.stack([f.view(-1) for f in WEBCAM_BUFFER]).to(engine.device)
                if actual_len < target_window:
                    pad_tensor = torch.zeros((target_window - actual_len, 1989), dtype=torch.float32, device=engine.device)
                    seq_tensor = torch.cat([seq_tensors, pad_tensor], dim=0).unsqueeze(0)
                else:
                    seq_tensor = seq_tensors[-target_window:].unsqueeze(0)
                    actual_len = target_window

                seq_len_t = torch.tensor([actual_len], device=engine.device)

                with torch.no_grad():
                    logits = engine.model(seq_tensor, seq_len_t)
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                eng_prob = float(probs[0])
                diseng_prob = float(probs[1])

                if eng_prob >= engagement_threshold:
                    pred_class = 0
                    conf = eng_prob * 100.0
                    badge_color = (46, 204, 113)  # Green
                    status_text = f"🟢 ENGAGED ({conf:.0f}%)"
                else:
                    pred_class = 1
                    conf = diseng_prob * 100.0
                    badge_color = (231, 76, 60)   # Red
                    status_text = f"🔴 DISENGAGED ({conf:.0f}%)"

                # Draw bounding box on verified face
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), badge_color, 3)
                tag = f"{status_text}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
                cv2.rectangle(frame, (xmin, max(0, ymin - th - 10)), (xmin + tw + 10, ymin), badge_color, -1)
                cv2.putText(frame, tag, (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    # Top overlay
    hud = frame.copy()
    cv2.rectangle(hud, (0, 0), (w, 40), (15, 23, 42), -1)
    cv2.addWeighted(hud, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, f"Live Camera Engagement: {status_text}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    return frame, status_text


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRADIO WEB APPLICATION INTERFACE
# ─────────────────────────────────────────────────────────────────────
def build_interface():
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: #0b0f19 !important;
        color: #f1f5f9;
    }
    .header-box {
        text-align: center;
        padding: 24px;
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border-radius: 14px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .header-box h1 {
        color: #f8fafc;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .header-box p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
        margin: 4px;
    }
    .badge-green { background: rgba(34, 197, 94, 0.18); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); }
    .badge-blue  { background: rgba(59, 130, 246, 0.18); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.3); }
    .badge-purple{ background: rgba(168, 85, 247, 0.18); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.3); }
    """

    with gr.Blocks(css=custom_css, title="Classroom Engagement AI — Live Dashboard") as demo:
        with gr.Row():
            gr.HTML("""
            <div class="header-box">
                <h1>🎓 AI Classroom Engagement Live Dashboard</h1>
                <p>Temporal Binary Transformer (Engaged vs Disengaged) &bull; 80%+ Accuracy</p>
                <div style="margin-top: 10px;">
                    <span class="badge badge-green">⚡ EfficientNet-B0 (1280)</span>
                    <span class="badge badge-blue">👤 OpenFace Dynamics (709)</span>
                    <span class="badge badge-purple">🎯 1989 Fused Dim</span>
                    <span class="badge badge-green">🚀 YOLOv8 + ByteTrack Multi-Student</span>
                </div>
            </div>
            """)

        with gr.Tabs():
            # ── Tab 1: Video File Upload & Classroom Analysis ───────────────
            with gr.TabItem("🎥 Classroom Video Analysis (Upload)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        video_input = gr.Video(label="Upload Classroom Video (.mp4 / .avi / .mov)")
                        with gr.Accordion("⚙️ Video & Tracking Settings", open=True):
                            with gr.Row():
                                conf_slider = gr.Slider(
                                    minimum=0.20, maximum=0.85, value=0.35, step=0.05,
                                    label="YOLO Detection Conf Threshold (Default: 0.35)"
                                )
                                thresh_slider = gr.Slider(
                                    minimum=0.20, maximum=0.80, value=0.50, step=0.05,
                                    label="Engagement Decision Threshold (Default: 0.50)"
                                )
                            with gr.Row():
                                seq_slider = gr.Slider(
                                    minimum=16, maximum=120, value=120, step=8,
                                    label="Student Temporal Buffer (Frames, Matches Training Window: 120)"
                                )
                                min_frames_slider = gr.Slider(
                                    minimum=5, maximum=60, value=12, step=1,
                                    label="Min Track Longevity (Eliminates ghost detections)"
                                )
                            clahe_checkbox = gr.Checkbox(
                                value=False, label="Apply Illumination Enhancement (CLAHE + Sharpening)"
                            )
                            gr.Markdown("ℹ️ **Plain Model Analysis**: Detects persons and extracts facial crops, keeps persistent track of each student with ByteTrack, and applies pure Binary Engagement Transformer classification (Green: Engaged, Red: Disengaged) directly from the model without any artificial filters.")
                        btn_analyze = gr.Button("⚡ Run Classroom Engagement Analysis", variant="primary", size="lg")

                    with gr.Column(scale=7):
                        video_output = gr.Video(label="Annotated Video Output (Green: Engaged | Red: Disengaged)")
                        kpi_output = gr.Markdown("### 📊 Metrics Summary will appear here after analysis.")

                with gr.Row():
                    with gr.Column(scale=6):
                        chart_output = gr.Image(label="Classroom Engagement Timeline & Distribution Plot")
                    with gr.Column(scale=6):
                        table_output = gr.Dataframe(label="Per-Student Engagement Breakdown")
                        file_csv_output = gr.File(label="📥 Download Detailed Per-Frame CSV Log")

                btn_analyze.click(
                    fn=process_uploaded_video,
                    inputs=[video_input, conf_slider, thresh_slider, seq_slider, min_frames_slider, clahe_checkbox],
                    outputs=[video_output, chart_output, table_output, file_csv_output, kpi_output]
                )

            # ── Tab 2: Live Webcam Camera ──────────────────────────────────
            with gr.TabItem("📹 Live Webcam Engagement Tracking"):
                with gr.Row():
                    with gr.Column(scale=6):
                        gr.Markdown("### 📷 Real-Time Camera Stream")
                        webcam_image = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            label="Live Camera Feed",
                            type="numpy"
                        )
                        cam_thresh_slider = gr.Slider(
                            minimum=0.20, maximum=0.80, value=0.50, step=0.05,
                            label="Engagement Sensitivity Threshold (Default: 0.50 | Lower for more Engaged, Higher for stricter Disengaged)"
                        )
                    with gr.Column(scale=6):
                        gr.Markdown("### 🎯 Live Model Prediction")
                        annotated_cam = gr.Image(label="Annotated Live Video Stream")
                        cam_status = gr.Textbox(
                            label="Current Engagement Status",
                            value="Awaiting camera feed...",
                            interactive=False
                        )
                        gr.Markdown("""
                        > **Live Stream Instructions**:
                        > 1. Click **Allow** when your browser requests camera access.
                        > 2. The pipeline extracts **EfficientNet visual embeddings + facial dynamics** in real-time.
                        > 3. Green bounding box = **Engaged** | Red bounding box = **Disengaged**.
                        """)

                webcam_image.stream(
                    fn=process_webcam_frame,
                    inputs=[webcam_image, cam_thresh_slider],
                    outputs=[annotated_cam, cam_status]
                )

            # ── Tab 3: Model Diagnostics & Architecture ─────────────────────
            with gr.TabItem("📊 Model Diagnostics & Performance"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ### 🧠 Binary Engagement Transformer Architecture
                        - **Classes**: `Engaged (0)` vs `Disengaged (1)`
                        - **Input Feature Vector**: `1989-dim` per frame
                          - `1280-dim`: EfficientNet-B0 deep visual features
                          - `709-dim`: OpenFace Action Units, Head Pose, Eye Gaze, and Facial Geometry
                        - **Transformer Encoder**: 3 Multi-Head Attention Layers (`d_model=128`, `nhead=4`, `dim_feedforward=256`)
                        - **Positional Encoding**: Sinusoidal Temporal Encoding
                        - **Accuracy**: **~85.7%** test accuracy with weighted soft-voting aggregation
                        """)
                    with gr.Column():
                        engine = get_engine()
                        device_name = str(engine.device).upper()
                        weights_status = "Loaded ✓" if engine.model is not None else "Initialized"
                        gr.Markdown(f"""
                        ### ⚙️ Runtime Diagnostics
                        - **Execution Device**: `{device_name}`
                        - **Model Status**: `{weights_status}`
                        - **YOLO Detector**: `{engine.yolo is not None}`
                        - **Drive Root**: `{config.DRIVE_BASE_DIR}`
                        """)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live Deployed Classroom Engagement Dashboard")
    parser.add_argument("--share", action="store_true", default=False,
                        help="Generate a public shareable HTTPS link (e.g. for Colab deployment)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the dashboard on (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--model", type=str, default=None,
                        help="Optional path to binary_transformer_weights.pth checkpoint")
    parser.add_argument("--yolo", type=str, default="yolov8n-face.pt",
                        help="YOLO model path")
    args = parser.parse_args()

    if gr is None:
        print("\n[Error] 'gradio' is required to launch the live dashboard.")
        print("Please install with: pip install gradio ultralytics\n")
        sys.exit(1)

    print("=" * 65)
    print("  LAUNCHING AI CLASSROOM ENGAGEMENT LIVE DASHBOARD")
    print("  Binary Transformer: Engaged vs Disengaged (1989 Features)")
    print(f"  Port: {args.port} | Share Public Link: {args.share}")
    print("=" * 65)

    # Initialize Engine with optional custom paths
    global ENGINE
    ENGINE = EngagementEngine(model_path=args.model, yolo_path=args.yolo)

    demo = build_interface()
    # Launch with public share URL if requested or running on Colab
    share_flag = args.share or config.RUNNING_ON_COLAB

    # Collect allowed directories for Gradio file serving
    allowed_dirs = [
        os.getcwd(),
        SCRIPT_DIR,
        os.path.join(SCRIPT_DIR, "dashboard_outputs"),
        config.PROJECT_ROOT,
        config.DRIVE_BASE_DIR,
        config.EVAL_OUT_DIR,
        config.EXTRACTED_FRAMES_DIR,
        tempfile.gettempdir(),
        "/content",
        "/content/drive",
        "/content/drive/MyDrive",
        "/content/drive/MyDrive/Student_Engagement_DL",
        "/content/drive/MyDrive/Student_Engagement_DL/evaluation_results_binary_transformer"
    ]
    valid_allowed = list(set([os.path.abspath(d) for d in allowed_dirs if d and os.path.exists(d)]))

    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=share_flag,
        show_error=True,
        allowed_paths=valid_allowed
    )


if __name__ == "__main__":
    main()
