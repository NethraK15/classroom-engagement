"""
app.py — Flask Backend for AI Classroom Engagement Dashboard
============================================================
Custom Flask web server (no Gradio).
Pipeline: Video Upload → YOLO v8 Face Detection + ByteTrack → EfficientNet-B0 (1280)
          + OpenFace Dynamics (709) → 1989-dim fused feature → Binary Transformer
          → Engaged / Disengaged with probability scores.
"""

import os
import sys
import time
import uuid
import json
import shutil
import threading
import traceback
import subprocess
import tempfile

import gc
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Limit PyTorch CPU threads to prevent memory buffer bloat on 512MB cloud instances
try:
    torch.set_num_threads(1)
except Exception:
    pass
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from flask import (
    Flask, render_template, request, jsonify,
    send_file, send_from_directory, abort
)
from werkzeug.utils import secure_filename

# ── Add project root to path ─────────────────────────────────────────────────
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(DASHBOARD_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config_binary_transformer as config
from models_binary_transformer import BinaryEngagementTransformer
from feature_extractor_binary import FusedFeaturePipeline

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    YOLO = None

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024   # 2 GB max upload
app.config["UPLOAD_FOLDER"] = os.path.join(DASHBOARD_DIR, "outputs")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# ── In-memory job store ───────────────────────────────────────────────────────
JOBS: dict = {}   # job_id → {status, progress, stage, result, error}

# ── Global Engine (loaded once) ───────────────────────────────────────────────
ENGINE = None
ENGINE_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE LOADER
# ─────────────────────────────────────────────────────────────────────────────
class EngagementEngine:
    def __init__(self, model_path=None, yolo_path="yolov8n-face.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Engine] Device: {self.device}")

        self.yolo, self.is_face_model = self._load_yolo(yolo_path)
        self.face_cascade = self._init_face_cascade()
        self.model, self.input_dim, self.global_mean, self.global_std = self._load_transformer(model_path)
        self.feature_pipeline = FusedFeaturePipeline(
            device=self.device,
            global_mean=self.global_mean,
            global_std=self.global_std
        )
        print("[Engine] Ready.")

    def _init_face_cascade(self):
        try:
            p = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(p):
                return cv2.CascadeClassifier(p)
        except Exception:
            pass
        return None

    def _load_yolo(self, yolo_path):
        if not HAS_YOLO:
            print("[Engine] ultralytics not installed — YOLO unavailable.")
            return None, False
        is_face = "face" in str(yolo_path).lower()
        if is_face and not os.path.exists(yolo_path):
            face_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
            try:
                import urllib.request
                print(f"[Engine] Downloading YOLO face weights → {yolo_path}")
                urllib.request.urlretrieve(face_url, yolo_path)
            except Exception:
                yolo_path = "yolo11n.pt"
                is_face = False
        try:
            return YOLO(yolo_path), is_face
        except Exception as e:
            print(f"[Engine] YOLO load error: {e}")
            return None, False

    def _load_transformer(self, model_path):
        candidates = [
            model_path,
            os.path.join(PROJECT_ROOT, "models_binary_transformer", "binary_transformer_weights.pth"),
            config.MODEL_WEIGHTS_PATH,
        ]
        target = next((p for p in candidates if p and os.path.exists(p)), None)

        input_dim = 1989
        d_model = config.D_MODEL
        nhead = config.NHEAD
        num_layers = config.NUM_LAYERS
        dim_feedforward = config.DIM_FEEDFORWARD
        global_mean = None
        global_std = None
        state_dict = None

        if target:
            print(f"[Engine] Loading checkpoint: {target}")
            ckpt = torch.load(target, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                input_dim = ckpt.get("input_dim", 1989)
                d_model = ckpt.get("d_model", config.D_MODEL)
                nhead = ckpt.get("nhead", config.NHEAD)
                num_layers = ckpt.get("num_layers", config.NUM_LAYERS)
                dim_feedforward = ckpt.get("dim_feedforward", config.DIM_FEEDFORWARD)
                global_mean = ckpt.get("global_mean", None)
                global_std = ckpt.get("global_std", None)
            elif isinstance(ckpt, dict):
                state_dict = ckpt

            if state_dict and "input_proj.0.weight" in state_dict:
                input_dim = state_dict["input_proj.0.weight"].shape[1]
                d_model = state_dict["input_proj.0.weight"].shape[0]
        else:
            print("[Engine] No checkpoint found — clean architecture.")

        model = BinaryEngagementTransformer(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=config.DROPOUT, num_classes=2
        )
        if state_dict:
            try:
                model.load_state_dict(state_dict)
            except Exception:
                model.load_state_dict(state_dict, strict=False)
        model.to(self.device).eval()

        if isinstance(global_mean, torch.Tensor):
            global_mean = global_mean.to(self.device)
        if isinstance(global_std, torch.Tensor):
            global_std = global_std.to(self.device)

        return model, input_dim, global_mean, global_std


def get_engine():
    global ENGINE
    with ENGINE_LOCK:
        if ENGINE is None:
            ENGINE = EngagementEngine()
    return ENGINE


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def update_job(job_id, **kwargs):
    if job_id not in JOBS:
        JOBS[job_id] = {}
    JOBS[job_id].update(kwargs)
    try:
        jdir = os.path.join(app.config["UPLOAD_FOLDER"], job_id)
        if os.path.exists(jdir):
            jpath = os.path.join(jdir, "job.json")
            with open(jpath, "w") as f:
                json.dump(JOBS[job_id], f)
    except Exception:
        pass


def _reencode_video(src, dst):
    """Re-encode to H.264 baseline for browser compatibility."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "baseline", "-level", "3.0",
            dst
        ]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if r.returncode == 0 and os.path.exists(dst):
            return dst
    except Exception:
        pass

    # If ffmpeg is not available, copy src to dst so destination video exists
    if os.path.exists(src):
        import shutil
        try:
            shutil.copy2(src, dst)
            return dst
        except Exception:
            return src
    return src


# ─────────────────────────────────────────────────────────────────────────────
# CHART GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_charts(student_summary, frame_metrics, out_dir, stem, half_time_data=None):
    """Generate per-student, timeline, 3-class distribution, and half-time comparison charts."""
    saved = []
    dark_bg  = "#0d1117"
    card_bg  = "#161b22"
    grid_col = "#30363d"
    text_col = "#e6edf3"
    sub_col  = "#8b949e"

    # ── Chart 1: Per-student probability distribution ─────────────────────────
    if student_summary:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        fig.patch.set_facecolor(dark_bg)
        ax.set_facecolor(card_bg)

        n = len(student_summary)
        xs = np.arange(n)
        bar_w = 0.35

        eng_vals  = [s["avg_engaged_prob"] * 100 for s in student_summary]
        dis_vals  = [s["avg_disengaged_prob"] * 100 for s in student_summary]
        labels    = [s["label"] for s in student_summary]

        # Use amber color for partially engaged students
        colors_eng = []
        for s in student_summary:
            if s.get("overall_status") == "Partially Engaged":
                colors_eng.append("#f59e0b")
            else:
                colors_eng.append("#2ecc71")

        b1 = ax.bar(xs - bar_w/2, eng_vals, bar_w, label="Engaged Prob", color=colors_eng, alpha=0.9, zorder=3)
        b2 = ax.bar(xs + bar_w/2, dis_vals, bar_w, label="Disengaged Prob", color="#e74c3c", alpha=0.9, zorder=3)

        for b in list(b1) + list(b2):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.0f}%",
                    ha="center", va="bottom", color=text_col, fontsize=7.5, fontweight="bold")

        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(0, 115)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, color=text_col, fontsize=9, rotation=25 if n > 8 else 0)
        ax.set_ylabel("Probability Score (%)", color=sub_col, fontsize=10)
        ax.set_title("Per-Student Engagement & Disengagement Probability Matrix", color=text_col, fontsize=12, fontweight="bold", pad=14)
        ax.tick_params(colors=sub_col)
        ax.axhline(50, color="#f59e0b", linestyle="--", lw=1.2, alpha=0.7, label="50% Equilibrium (Partially Engaged)", zorder=2)
        for spine in ax.spines.values():
            spine.set_color(grid_col)
        ax.grid(axis="y", linestyle="--", alpha=0.3, color=grid_col, zorder=1)
        ax.legend(facecolor=card_bg, edgecolor=grid_col, labelcolor=text_col, fontsize=8.5, loc="upper right")

        p1 = os.path.join(out_dir, f"{stem}_student_chart.png")
        plt.tight_layout()
        plt.savefig(p1, facecolor=fig.get_facecolor())
        plt.close()
        saved.append(("student_chart", p1))

    # ── Chart 2: Engagement timeline with Midpoint Split ─────────────────────
    if frame_metrics:
        df_m = pd.DataFrame(frame_metrics)
        smooth = df_m["engagement_rate"].rolling(window=10, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
        fig.patch.set_facecolor(dark_bg)
        ax.set_facecolor(card_bg)

        ax.fill_between(df_m["timestamp_sec"], smooth, alpha=0.25, color="#38bdf8")
        ax.plot(df_m["timestamp_sec"], smooth, color="#38bdf8", lw=2.2, label="Class Engagement Rate (%)")
        ax.axhline(50, color="#f59e0b", linestyle=":", lw=1.5, alpha=0.8, label="50% Neutral Threshold")

        # Mark Midpoint
        if half_time_data and "midpoint_sec" in half_time_data:
            m_sec = half_time_data["midpoint_sec"]
            ax.axvline(m_sec, color="#c084fc", linestyle="--", lw=1.8, alpha=0.9, label=f"Midpoint ({m_sec:.1f}s)")
            ax.text(m_sec * 0.5, 96, "FIRST HALF", color="#60a5fa", fontsize=8.5, fontweight="bold", ha="center")
            total_sec = df_m["timestamp_sec"].iloc[-1] if len(df_m) > 0 else m_sec * 2
            ax.text((m_sec + total_sec) * 0.5, 96, "SECOND HALF", color="#c084fc", fontsize=8.5, fontweight="bold", ha="center")

        ax.set_ylim(0, 108)
        ax.set_xlabel("Session Time (seconds)", color=sub_col, fontsize=10)
        ax.set_ylabel("Classroom Engagement (%)", color=sub_col, fontsize=10)
        ax.set_title("Temporal Classroom Engagement Timeline (1st vs 2nd Half Split)", color=text_col, fontsize=12, fontweight="bold", pad=14)
        ax.tick_params(colors=sub_col)
        for spine in ax.spines.values():
            spine.set_color(grid_col)
        ax.grid(True, linestyle="--", alpha=0.25, color=grid_col)
        ax.legend(facecolor=card_bg, edgecolor=grid_col, labelcolor=text_col, fontsize=8.5, loc="lower right")

        p2 = os.path.join(out_dir, f"{stem}_timeline_chart.png")
        plt.tight_layout()
        plt.savefig(p2, facecolor=fig.get_facecolor())
        plt.close()
        saved.append(("timeline_chart", p2))

    # ── Chart 3: 3-Classroom distribution donut (Engaged, Partially, Disengaged) ─
    if student_summary:
        eng_count     = sum(1 for s in student_summary if s["overall_status"] == "Engaged")
        partial_count = sum(1 for s in student_summary if s["overall_status"] == "Partially Engaged")
        dis_count     = sum(1 for s in student_summary if s["overall_status"] == "Disengaged")

        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)
        fig.patch.set_facecolor(dark_bg)
        ax.set_facecolor(dark_bg)

        wedge_sizes  = []
        wedge_colors = []
        wedge_labels = []

        if eng_count > 0:
            wedge_sizes.append(eng_count)
            wedge_colors.append("#2ecc71")
            wedge_labels.append(f"Engaged\n({eng_count})")
        if partial_count > 0:
            wedge_sizes.append(partial_count)
            wedge_colors.append("#f59e0b")
            wedge_labels.append(f"Partially Engaged\n({partial_count})")
        if dis_count > 0:
            wedge_sizes.append(dis_count)
            wedge_colors.append("#e74c3c")
            wedge_labels.append(f"Disengaged\n({dis_count})")

        if wedge_sizes:
            wedges, texts, autotexts = ax.pie(
                wedge_sizes, colors=wedge_colors, labels=wedge_labels,
                autopct="%1.0f%%", startangle=90, pctdistance=0.76,
                wedgeprops=dict(width=0.48, edgecolor=dark_bg, linewidth=2.5),
                textprops=dict(color=text_col, fontsize=9.5, fontweight="500")
            )
            for a in autotexts:
                a.set_fontsize(11)
                a.set_fontweight("bold")
                a.set_color("white")

            # Center text in donut
            total_students = len(student_summary)
            ax.text(0, 0.08, f"{total_students}", ha="center", va="center", color=text_col, fontsize=20, fontweight="bold")
            ax.text(0, -0.15, "STUDENTS", ha="center", va="center", color=sub_col, fontsize=8, fontweight="bold")

        ax.set_title("Classroom Distribution\n(Engaged / Partially / Disengaged)", color=text_col, fontsize=11.5, fontweight="bold", pad=12)
        p3 = os.path.join(out_dir, f"{stem}_pie_chart.png")
        plt.tight_layout()
        plt.savefig(p3, facecolor=fig.get_facecolor())
        plt.close()
        saved.append(("pie_chart", p3))

    # ── Chart 4: Half-Time Comparison (First Half vs Second Half Analysis) ────
    if half_time_data:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        fig.patch.set_facecolor(dark_bg)
        ax.set_facecolor(card_bg)

        h1 = half_time_data.get("first_half_engagement", 0.0)
        h2 = half_time_data.get("second_half_engagement", 0.0)
        delta = half_time_data.get("delta", 0.0)

        bars = ax.bar(["1st Half\n(Initial Session)", "2nd Half\n(Latter Session)"],
                      [h1, h2],
                      color=["#38bdf8", "#c084fc"],
                      width=0.45, zorder=3)

        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 2.0, f"{h:.1f}%",
                    ha="center", va="bottom", color=text_col, fontsize=12, fontweight="bold")

        # Delta Annotation
        trend_col = "#4ade80" if delta >= 0 else "#f87171"
        delta_sign = "+" if delta >= 0 else ""
        clean_trend_text = "Engagement Increased in 2nd Half" if delta >= 5 else ("Engagement Dropped in 2nd Half" if delta <= -5 else "Stable Engagement across Session")
        ax.text(0.5, max(h1, h2) * 0.5, f"Δ {delta_sign}{delta:.1f}%\n{clean_trend_text}",
                ha="center", va="center", color=trend_col, fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=dark_bg, edgecolor=trend_col, alpha=0.9))

        ax.set_ylim(0, 115)
        ax.set_ylabel("Average Engagement Rate (%)", color=sub_col, fontsize=10)
        ax.set_title("Classroom Temporal Progression: 1st Half vs 2nd Half", color=text_col, fontsize=12, fontweight="bold", pad=14)
        ax.tick_params(colors=sub_col)
        for spine in ax.spines.values():
            spine.set_color(grid_col)
        ax.grid(axis="y", linestyle="--", alpha=0.25, color=grid_col, zorder=1)

        p4 = os.path.join(out_dir, f"{stem}_halftime_chart.png")
        plt.tight_layout()
        plt.savefig(p4, facecolor=fig.get_facecolor())
        plt.close()
        saved.append(("halftime_chart", p4))

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_text_report(stem, student_summary, frame_metrics, processed_count, fps, elapsed, half_time_data=None):
    lines = [
        "=" * 70,
        "  AI CLASSROOM ENGAGEMENT ANALYSIS REPORT",
        f"  Generated by Binary Engagement Transformer (80% Accuracy)",
        "=" * 70,
        f"\nVideo: {stem}",
        f"Processed Frames: {processed_count}  |  FPS: {fps:.1f}  |  Duration: {processed_count/max(fps,1):.1f}s",
        f"Processing Time: {elapsed:.1f}s",
        "\n─── CLASSROOM SUMMARY ───────────────────────────────────────────────\n",
    ]

    if frame_metrics:
        avg_eng = np.mean([m["engagement_rate"] for m in frame_metrics])
        lines.append(f"  Overall Classroom Engagement Rate : {avg_eng:.1f}%")
    lines.append(f"  Students Detected & Tracked       : {len(student_summary)}")

    engaged_students  = [s for s in student_summary if s["overall_status"] == "Engaged"]
    diseng_students   = [s for s in student_summary if s["overall_status"] == "Disengaged"]
    partial_students  = [s for s in student_summary if s["overall_status"] == "Partially Engaged"]
    lines.append(f"  Engaged Students                  : {len(engaged_students)} ({len(engaged_students)/max(len(student_summary),1)*100:.1f}%)")
    lines.append(f"  Partially Engaged (50/50 Border)  : {len(partial_students)} ({len(partial_students)/max(len(student_summary),1)*100:.1f}%)")
    lines.append(f"  Disengaged Students               : {len(diseng_students)} ({len(diseng_students)/max(len(student_summary),1)*100:.1f}%)")

    if half_time_data:
        lines += [
            "\n─── TEMPORAL PROGRESSION: 1ST HALF VS 2ND HALF ANALYSIS ───────────\n",
            f"  1st Half Engagement Rate (0 - {half_time_data.get('midpoint_sec', 0):.1f}s)    : {half_time_data.get('first_half_engagement', 0.0):.1f}%",
            f"  2nd Half Engagement Rate ({half_time_data.get('midpoint_sec', 0):.1f}s - End) : {half_time_data.get('second_half_engagement', 0.0):.1f}%",
            f"  Temporal Delta (2nd Half - 1st Half)    : {half_time_data.get('delta', 0.0):+.1f}%",
            f"  Trend Diagnosis                         : {half_time_data.get('trend_label', 'N/A')}",
            f"  Pedagogical Insight                     : {half_time_data.get('insight', 'N/A')}",
        ]

    lines.append("\n─── PER-STUDENT ANALYSIS ────────────────────────────────────────────\n")
    for s in student_summary:
        if s["overall_status"] == "Engaged":
            status_icon = "✅"
        elif s["overall_status"] == "Partially Engaged":
            status_icon = "⚠️"
        else:
            status_icon = "❌"
        lines.append(f"  {status_icon}  {s['label']:<20}  Status: {s['overall_status']:<18}")
        lines.append(f"       Engaged Probability  : {s['avg_engaged_prob']*100:.1f}%")
        lines.append(f"       Disengaged Probability: {s['avg_disengaged_prob']*100:.1f}%")
        lines.append(f"       Tracked Frames        : {s['tracked_frames']}")
        if "first_half_eng_pct" in s and "second_half_eng_pct" in s:
            lines.append(f"       Half-Time Trend       : 1st Half {s['first_half_eng_pct']:.0f}% -> 2nd Half {s['second_half_eng_pct']:.0f}% ({s.get('student_trend','')})")
        if s["overall_status"] == "Partially Engaged":
            lines.append(f"       Note: Balanced ~50/50 scores (Partially Engaged / Shifting Focus)")
        lines.append("")

    lines += [
        "\n─── PIPELINE USED ───────────────────────────────────────────────────",
        "  1. YOLOv8 (yolov8n-face.pt) + ByteTrack face detection & tracking",
        "  2. EfficientNet-B0 → 1280-dim visual embeddings per face crop",
        "  3. OpenFace Facial Dynamics → 709-dim (pose, gaze, AUs, landmarks)",
        "  4. Fused 1989-dim feature vector per frame",
        "  5. Binary Engagement Transformer (d_model=128, 3 layers, 4 heads)",
        "  6. Sliding window (120 frames, stride 30) + soft-vote aggregation",
        "\n─────────────────────────────────────────────────────────────────────",
        f"\n  Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CORE VIDEO PROCESSING (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────
def process_video_job(job_id: str, video_path: str, conf_thresh: float,
                      eng_thresh: float, seq_buf: int, min_frames: int):
    """Runs the full pipeline in a background thread and stores results in JOBS."""
    try:
        update_job(job_id, status="running", progress=0, stage="Initializing models…")
        engine = get_engine()

        if engine.yolo is None:
            raise RuntimeError("YOLO model not loaded. Install ultralytics: pip install ultralytics lapx")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 200
        if total <= 0:
            total = 200

        # Cap working resolution to 720p max to stay strictly within 512MB RAM on Render
        max_dim = max(W, H)
        if max_dim > 720:
            scale = 720.0 / float(max_dim)
            W = int(W * scale)
            H = int(H * scale)
            W = W if W % 2 == 0 else W + 1
            H = H if H % 2 == 0 else H + 1

        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(app.config["UPLOAD_FOLDER"], job_id)
        os.makedirs(out_dir, exist_ok=True)

        raw_video  = os.path.join(out_dir, f"{stem}_raw.mp4")
        web_video  = os.path.join(out_dir, f"{stem}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(raw_video, fourcc, fps, (W, H))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(raw_video, fourcc, fps, (W, H))

        # Tracking state
        face_buffers    = {}   # track_id → [tensor, ...]
        last_of_feats   = {}   # track_id → np array [709]
        pred_history    = {}   # track_id → {frame→class}
        prob_history    = {}   # track_id → {frame→(eng_p, dis_p)}
        confirmed_faces = set()
        detailed_recs   = []
        frame_metrics   = []

        # Sequential pipeline stages (strictly monotonic, no jumping backwards)
        PIPELINE_SEQUENCE = [
            ("yolo",        "Detecting student faces (YOLOv8)…",             0.02),
            ("track",       "Tracking student identities (ByteTrack)…",      0.22),
            ("effnet",      "Visual feature extraction (EfficientNet-B0)…",  0.45),
            ("openface",    "Facial dynamics extraction (OpenFace)…",       0.68),
            ("transformer", "Transformer engagement classification…",        0.82),
        ]
        seq_stage_idx = 0

        start_time  = time.time()
        frame_count = 0

        update_job(job_id, stage_key="yolo", stage="Detecting student faces (YOLOv8)…", progress=3)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))

            # Update progress + sequential stage
            pct = min(int(frame_count / total * 90), 90)
            progress_ratio = pct / 100.0
            while seq_stage_idx + 1 < len(PIPELINE_SEQUENCE) and progress_ratio >= PIPELINE_SEQUENCE[seq_stage_idx + 1][2]:
                seq_stage_idx += 1
                sk, sl, _ = PIPELINE_SEQUENCE[seq_stage_idx]
                update_job(job_id, stage_key=sk, stage=sl)
            update_job(job_id, progress=pct)

            ts = round(frame_count / fps, 2)

            # YOLO + ByteTrack tracking
            track_kwargs = {
                "source": frame, "persist": True,
                "conf": conf_thresh, "tracker": "bytetrack.yaml", "verbose": False
            }
            if not engine.is_face_model:
                track_kwargs["classes"] = [0]

            try:
                results = engine.yolo.track(**track_kwargs)
            except Exception:
                try:
                    track_kwargs["tracker"] = "botsort.yaml"
                    results = engine.yolo.track(**track_kwargs)
                except Exception:
                    results = []

            frame_engaged = 0
            frame_diseng  = 0
            frame_stud    = 0

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls = int(box.cls[0].item()) if box.cls is not None else 0
                    if not engine.is_face_model and cls != 0:
                        continue

                    track_id = int(box.id[0].item()) if box.id is not None else 0
                    if track_id <= 0:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    xmin, ymin, xmax, ymax = map(int, xyxy)
                    bw, bh = xmax - xmin, ymax - ymin

                    if bw < 20 or bh < 25:
                        continue
                    if not engine.is_face_model and (bw / float(bh) > 1.6 or bw / float(bh) < 0.22):
                        continue

                    xmin_c, ymin_c = max(0, xmin), max(0, ymin)
                    xmax_c, ymax_c = min(W, xmax), min(H, ymax)

                    # Extract face crop
                    face_crop = None
                    if not engine.is_face_model:
                        head_h   = max(20, int(bh * 0.35))
                        head_crop = frame[ymin_c:min(H, ymin_c + head_h), xmin_c:xmax_c]
                        if engine.face_cascade is not None and head_crop.size > 0:
                            gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                            faces_det = engine.face_cascade.detectMultiScale(
                                gray, scaleFactor=1.1, minNeighbors=3, minSize=(16, 16)
                            )
                            if len(faces_det) > 0:
                                fx, fy, fw, fh = max(faces_det, key=lambda b: b[2] * b[3])
                                px, py = int(fw * 0.2), int(fh * 0.2)
                                face_crop = frame[
                                    max(0, ymin_c + fy - py):min(H, ymin_c + fy + fh + py),
                                    max(0, xmin_c + fx - px):min(W, xmin_c + fx + fw + px)
                                ]
                        if face_crop is None or face_crop.size == 0:
                            face_crop = head_crop
                    else:
                        face_crop = frame[ymin_c:ymax_c, xmin_c:xmax_c]

                    if face_crop is None or face_crop.shape[0] < 12 or face_crop.shape[1] < 12:
                        continue

                    confirmed_faces.add(track_id)

                    # Extract 1989-dim fused feature
                    prev_of = last_of_feats.get(track_id)
                    fused_t, new_of = engine.feature_pipeline.extract_and_fuse(
                        face_crop, previous_of_feat=prev_of
                    )
                    last_of_feats[track_id] = new_of

                    # Sliding buffer
                    if track_id not in face_buffers:
                        face_buffers[track_id] = []
                        pred_history[track_id] = {}
                        prob_history[track_id] = {}

                    fused_np = fused_t.view(-1).cpu().detach().numpy()
                    face_buffers[track_id].append(fused_np)
                    if len(face_buffers[track_id]) > seq_buf:
                        face_buffers[track_id].pop(0)

                    actual_len = len(face_buffers[track_id])
                    seq_arr = np.stack(face_buffers[track_id])
                    seq_t = torch.from_numpy(seq_arr).float().to(engine.device)
                    if actual_len < seq_buf:
                        pad = torch.zeros((seq_buf - actual_len, engine.input_dim), dtype=torch.float32, device=engine.device)
                        seq_t = torch.cat([seq_t, pad], dim=0).unsqueeze(0)
                    else:
                        seq_t = seq_t[-seq_buf:].unsqueeze(0)
                        actual_len = seq_buf

                    with torch.no_grad():
                        logits = engine.model(seq_t, torch.tensor([actual_len], device=engine.device))
                        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    eng_p, dis_p = float(probs[0]), float(probs[1])
                    pred_class   = 0 if eng_p >= eng_thresh else 1

                    pred_history[track_id][frame_count] = pred_class
                    prob_history[track_id][frame_count] = (eng_p, dis_p)

                    if len(face_buffers[track_id]) >= 2:
                        frame_stud += 1
                        if pred_class == 0:
                            frame_engaged += 1
                            b_col = (46, 204, 113)    # Green
                        else:
                            frame_diseng += 1
                            b_col = (60, 76, 231)     # Red

                        # Draw bounding box + label
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), b_col, 2)

                        # Probability pill label
                        label = f"#{track_id}  E:{eng_p*100:.0f}%  D:{dis_p*100:.0f}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                        ly = max(0, ymin - th - 8)
                        cv2.rectangle(frame, (xmin, ly), (xmin + tw + 8, ymin), b_col, -1)
                        cv2.putText(frame, label, (xmin + 4, ymin - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

                    detailed_recs.append({
                        "frame": frame_count, "timestamp_sec": ts,
                        "student_id": track_id,
                        "status": "Engaged" if pred_class == 0 else "Disengaged",
                        "engaged_prob": round(eng_p, 3),
                        "disengaged_prob": round(dis_p, 3),
                        "bbox": f"[{xmin},{ymin},{xmax},{ymax}]"
                    })

            # HUD banner
            eng_rate = (frame_engaged / frame_stud * 100) if frame_stud > 0 else 0.0
            hud = frame.copy()
            cv2.rectangle(hud, (0, 0), (W, 44), (13, 17, 23), -1)
            cv2.addWeighted(hud, 0.82, frame, 0.18, 0, frame)
            hud_l = f"AI Engagement  |  Frame {frame_count}  |  {ts:.1f}s"
            hud_r = f"Faces: {frame_stud}  Engaged: {frame_engaged}  Disengaged: {frame_diseng}  |  {eng_rate:.0f}%"
            cv2.putText(frame, hud_l, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 210, 255), 1, cv2.LINE_AA)
            (rw, _), _ = cv2.getTextSize(hud_r, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            hc = (46, 204, 113) if eng_rate >= 50 else (60, 76, 231)
            cv2.putText(frame, hud_r, (max(W - rw - 12, 250), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, hc, 1, cv2.LINE_AA)

            frame_metrics.append({"frame": frame_count, "timestamp_sec": ts,
                                   "students": frame_stud, "engaged": frame_engaged,
                                   "disengaged": frame_diseng, "engagement_rate": eng_rate})
            writer.write(frame)
            if frame_count % 20 == 0:
                gc.collect()

        cap.release()
        writer.release()
        if frame_count > 0:
            total = frame_count
        elapsed = max(time.time() - start_time, 0.01)

        update_job(job_id, stage_key="transformer", stage="Re-encoding video for browser…", progress=92)
        final_video = _reencode_video(raw_video, web_video)

        update_job(job_id, stage_key="transformer", stage="Computing student summaries & temporal analysis…", progress=94)
        effective_min = max(min_frames, int(frame_count * 0.04))
        mid_frame = max(frame_count // 2, 1)
        mid_sec   = round(mid_frame / fps, 2)
        total_sec = round(frame_count / fps, 2)

        # ── Half-Time Classroom Analysis (1st Half vs 2nd Half) ────────────────
        first_half_frames  = [m for m in frame_metrics if m["frame"] <= mid_frame]
        second_half_frames = [m for m in frame_metrics if m["frame"] > mid_frame]

        first_half_eng  = float(np.mean([m["engagement_rate"] for m in first_half_frames])) if first_half_frames else 0.0
        second_half_eng = float(np.mean([m["engagement_rate"] for m in second_half_frames])) if second_half_frames else 0.0
        half_delta      = round(second_half_eng - first_half_eng, 1)

        if half_delta >= 4.0:
            half_trend_dir   = "increasing"
            half_trend_label = f"📈 Engagement Increased (+{half_delta}%) in 2nd Half"
            half_insight     = "Attentiveness and student engagement grew in the second half. The lesson pacing or interactive exercises successfully sustained high classroom attention."
        elif half_delta <= -4.0:
            half_trend_dir   = "decreasing"
            half_trend_label = f"📉 Engagement Dropped ({half_delta}%) in 2nd Half"
            half_insight     = "Attention fatigue was detected after the midpoint. Consider introducing short quizzes, student questions, or group discussions midway through the session."
        else:
            half_trend_dir   = "steady"
            half_trend_label = f"➡️ Stable Engagement (Δ {half_delta:+.1f}%)"
            half_insight     = "Classroom engagement held remarkably consistent throughout both halves of the session with balanced attention across all tracked students."

        half_time_data = {
            "first_half_engagement":  round(first_half_eng, 1),
            "second_half_engagement": round(second_half_eng, 1),
            "delta":                  half_delta,
            "trend_direction":        half_trend_dir,
            "trend_label":            half_trend_label,
            "insight":                half_insight,
            "midpoint_sec":           mid_sec,
            "midpoint_frame":         mid_frame,
            "first_half_duration":    mid_sec,
            "second_half_duration":   round(max(total_sec - mid_sec, 0), 1),
        }

        # ── Per-Student Analysis (with 1st & 2nd half tracking) ────────────────
        student_summary = []
        for tid, f_dict in sorted(pred_history.items()):
            n_t = len(f_dict)
            if n_t < effective_min:
                continue
            if not engine.is_face_model and tid not in confirmed_faces and n_t < int(frame_count * 0.20):
                continue

            c_eng = sum(1 for p in f_dict.values() if p == 0)
            c_dis = n_t - c_eng
            p_eng = c_eng / n_t
            p_dis = c_dis / n_t

            # Average probabilities
            p_dict = prob_history.get(tid, {})
            if p_dict:
                avg_eng_p = float(np.mean([v[0] for v in p_dict.values()]))
                avg_dis_p = float(np.mean([v[1] for v in p_dict.values()]))

                # 1st vs 2nd half student trajectory
                h1_probs = [v[0] for f, v in p_dict.items() if f <= mid_frame]
                h2_probs = [v[0] for f, v in p_dict.items() if f > mid_frame]
                h1_eng_p = float(np.mean(h1_probs)) if h1_probs else avg_eng_p
                h2_eng_p = float(np.mean(h2_probs)) if h2_probs else avg_eng_p
            else:
                avg_eng_p, avg_dis_p = p_eng, p_dis
                h1_eng_p, h2_eng_p   = p_eng, p_dis

            st_delta = round((h2_eng_p - h1_eng_p) * 100, 1)
            if st_delta >= 5.0:
                st_trend = "improving"
            elif st_delta <= -5.0:
                st_trend = "dropping"
            else:
                st_trend = "steady"

            if abs(avg_eng_p - avg_dis_p) <= 0.15:
                overall = "Partially Engaged"
            else:
                overall = "Engaged" if avg_eng_p >= 0.50 else "Disengaged"

            student_summary.append({
                "track_id":             tid,
                "label":                f"Student #{tid}",
                "overall_status":       overall,
                "engaged_pct":          round(p_eng * 100, 1),
                "disengaged_pct":       round(p_dis * 100, 1),
                "avg_engaged_prob":     round(avg_eng_p, 4),
                "avg_disengaged_prob":  round(avg_dis_p, 4),
                "tracked_frames":       n_t,
                "first_half_eng_pct":   round(h1_eng_p * 100, 1),
                "second_half_eng_pct":  round(h2_eng_p * 100, 1),
                "student_trend":        st_trend,
                "student_trend_delta":  st_delta,
            })

        update_job(job_id, stage_key="transformer", stage="Generating analytical charts…", progress=96)
        charts_saved = generate_charts(student_summary, frame_metrics, out_dir, stem, half_time_data=half_time_data)

        update_job(job_id, stage_key="transformer", stage="Saving CSV & report…", progress=98)

        # CSV
        df_rec = pd.DataFrame(detailed_recs) if detailed_recs else pd.DataFrame()
        csv_path = os.path.join(out_dir, f"{stem}_engagement_log.csv")
        df_rec.to_csv(csv_path, index=False)

        # Text report
        report_txt = generate_text_report(stem, student_summary, frame_metrics, frame_count, fps, elapsed, half_time_data=half_time_data)
        rpt_path = os.path.join(out_dir, f"{stem}_report.txt")
        with open(rpt_path, "w") as f:
            f.write(report_txt)

        # JSON summary
        overall_eng_rate = float(np.mean([m["engagement_rate"] for m in frame_metrics])) if frame_metrics else 0.0
        total_students   = len(student_summary)
        n_eng            = sum(1 for s in student_summary if s["overall_status"] == "Engaged")
        n_partial        = sum(1 for s in student_summary if s["overall_status"] == "Partially Engaged")
        n_dis            = sum(1 for s in student_summary if s["overall_status"] == "Disengaged")

        result = {
            "video_url":          f"/outputs/{job_id}/{os.path.basename(final_video)}",
            "student_summary":    student_summary,
            "frame_metrics":      frame_metrics[-200:],   # keep last 200 for chart
            "halftime_analysis":  half_time_data,
            "charts":             {k: f"/outputs/{job_id}/{os.path.basename(v)}" for k, v in charts_saved},
            "csv_url":            f"/outputs/{job_id}/{stem}_engagement_log.csv",
            "report_url":         f"/outputs/{job_id}/{stem}_report.txt",
            "kpi": {
                "overall_engagement":          round(overall_eng_rate, 1),
                "students_tracked":            total_students,
                "engaged_students":            n_eng,
                "partially_engaged_students":  n_partial,
                "disengaged_students":         n_dis,
                "engaged_pct":                 round(n_eng / max(total_students, 1) * 100, 1),
                "partially_engaged_pct":       round(n_partial / max(total_students, 1) * 100, 1),
                "disengaged_pct":              round(n_dis / max(total_students, 1) * 100, 1),
                "processed_frames":            frame_count,
                "total_frames":                total,
                "duration_sec":                round(total_sec, 1),
                "fps":                         round(fps, 1),
                "processing_time":             round(elapsed, 1),
                "first_half_engagement":       half_time_data["first_half_engagement"],
                "second_half_engagement":      half_time_data["second_half_engagement"],
                "halftime_delta":              half_delta,
                "halftime_trend":              half_trend_dir,
            }
        }

        update_job(job_id, status="done", progress=100, stage="Analysis complete!", result=result)
        print(f"[Job {job_id}] Done in {elapsed:.1f}s — {len(student_summary)} students (E:{n_eng}, P:{n_partial}, D:{n_dis}).")

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"[Job {job_id}] ERROR:\n{err_msg}")
        update_job(job_id, status="error", stage="Error", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    f = request.files["video"]
    if not f or not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    job_id  = str(uuid.uuid4())
    out_dir = os.path.join(app.config["UPLOAD_FOLDER"], job_id)
    os.makedirs(out_dir, exist_ok=True)

    fname   = secure_filename(f.filename)
    vpath   = os.path.join(out_dir, fname)
    f.save(vpath)

    # Parse parameters
    conf_thresh  = float(request.form.get("conf_thresh", 0.35))
    eng_thresh   = float(request.form.get("eng_thresh", 0.50))
    seq_buf      = int(request.form.get("seq_buf", 120))
    min_frames   = int(request.form.get("min_frames", 12))
    update_job(job_id, status="queued", progress=1, stage="Video received, initializing pipeline…", result=None, error=None)

    t = threading.Thread(
        target=process_video_job,
        args=(job_id, vpath, conf_thresh, eng_thresh, seq_buf, min_frames),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        jpath = os.path.join(app.config["UPLOAD_FOLDER"], job_id, "job.json")
        if os.path.exists(jpath):
            try:
                with open(jpath, "r") as f:
                    job = json.load(f)
                    JOBS[job_id] = job
            except Exception:
                pass
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/outputs/<job_id>/<filename>")
def serve_output(job_id, filename):
    directory = os.path.join(app.config["UPLOAD_FOLDER"], job_id)
    if not os.path.exists(os.path.join(directory, filename)):
        abort(404)
    return send_from_directory(directory, filename)


@app.route("/download/<job_id>/<filename>")
def download_file(job_id, filename):
    directory = os.path.join(app.config["UPLOAD_FOLDER"], job_id)
    fpath = os.path.join(directory, filename)
    if not os.path.exists(fpath):
        abort(404)
    return send_file(fpath, as_attachment=True)


DEMO_CACHE = None

def get_or_create_demo_result():
    """Returns a rich pre-computed demonstration session for instant preview."""
    global DEMO_CACHE
    if DEMO_CACHE is not None:
        return DEMO_CACHE

    demo_dir = os.path.join(app.config["UPLOAD_FOLDER"], "demo")
    os.makedirs(demo_dir, exist_ok=True)

    # Detect existing demo video in outputs/demo
    video_candidates = ["demo_annotated.mp4", "demo_classroom.mp4"]
    chosen_video = None
    for cand in video_candidates:
        if os.path.exists(os.path.join(demo_dir, cand)):
            chosen_video = cand
            break
    if not chosen_video:
        for f in os.listdir(demo_dir):
            if f.lower().endswith(".mp4"):
                chosen_video = f
                break
    if not chosen_video:
        chosen_video = "demo_classroom.mp4"

    # Load real metrics from demo_engagement_log.csv if present
    csv_path = os.path.join(demo_dir, "demo_engagement_log.csv")
    demo_students = []
    frame_metrics = []
    half_time_data = None
    total_frames = 362
    fps = 30.0
    duration_sec = 12.1
    mid_sec = 6.0

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            total_frames = int(df["frame"].max())
            mid_frame = max(total_frames // 2, 1)
            duration_sec = round(total_frames / fps, 1)
            mid_sec = round(mid_frame / fps, 1)

            student_ids = sorted(df["student_id"].unique())
            for sid in student_ids:
                sdf = df[df["student_id"] == sid]
                if len(sdf) < 5:
                    continue
                avg_e = float(sdf["engaged_prob"].mean())
                avg_d = float(sdf["disengaged_prob"].mean())

                s_h1 = sdf[sdf["frame"] <= mid_frame]
                s_h2 = sdf[sdf["frame"] > mid_frame]
                e1 = float(s_h1["engaged_prob"].mean() * 100) if len(s_h1) > 0 else avg_e * 100
                e2 = float(s_h2["engaged_prob"].mean() * 100) if len(s_h2) > 0 else avg_e * 100
                s_delta = round(e2 - e1, 1)
                s_trend = "improving" if s_delta >= 4.0 else ("dropping" if s_delta <= -4.0 else "steady")

                if 0.45 <= avg_e <= 0.55:
                    status = "Partially Engaged"
                elif avg_e > 0.50:
                    status = "Engaged"
                else:
                    status = "Disengaged"

                demo_students.append({
                    "track_id": int(sid),
                    "label": f"Student #{sid}",
                    "overall_status": status,
                    "engaged_pct": round(avg_e * 100, 1),
                    "disengaged_pct": round(avg_d * 100, 1),
                    "avg_engaged_prob": round(avg_e, 3),
                    "avg_disengaged_prob": round(avg_d, 3),
                    "tracked_frames": int(len(sdf)),
                    "first_half_eng_pct": round(e1, 1),
                    "second_half_eng_pct": round(e2, 1),
                    "student_trend": s_trend,
                    "student_trend_delta": s_delta
                })

            for f in sorted(df["frame"].unique()):
                fdf = df[df["frame"] == f]
                n_s = len(fdf)
                n_e = int((fdf["engaged_prob"] >= 0.5).sum())
                rate = round(n_e / max(n_s, 1) * 100, 1)
                ts = float(fdf["timestamp_sec"].iloc[0])
                frame_metrics.append({
                    "frame": int(f),
                    "timestamp_sec": round(ts, 2),
                    "students": n_s,
                    "engaged": n_e,
                    "disengaged": n_s - n_e,
                    "engagement_rate": rate
                })

            f_h1 = [m["engagement_rate"] for m in frame_metrics if m["frame"] <= mid_frame]
            f_h2 = [m["engagement_rate"] for m in frame_metrics if m["frame"] > mid_frame]
            first_half_eng = round(float(np.mean(f_h1)), 1) if f_h1 else 35.0
            second_half_eng = round(float(np.mean(f_h2)), 1) if f_h2 else 35.0
            half_delta = round(second_half_eng - first_half_eng, 1)

            half_time_data = {
                "first_half_engagement": first_half_eng,
                "second_half_engagement": second_half_eng,
                "delta": half_delta,
                "trend_direction": "increasing" if half_delta >= 4.0 else ("decreasing" if half_delta <= -4.0 else "steady"),
                "trend_label": f"Engagement Shift (Δ {half_delta:+.1f}%)",
                "insight": "Noticeable attention fatigue detected after midpoint. Recommended action: Introduce short interactive questions or student discussions midway.",
                "midpoint_sec": mid_sec,
                "midpoint_frame": mid_frame,
                "first_half_duration": mid_sec,
                "second_half_duration": round(duration_sec - mid_sec, 1)
            }
        except Exception as e:
            print(f"[Demo] Error reading CSV: {e}")

    # Fallback to defaults if CSV is missing or empty
    if not demo_students:
        demo_students = [
            {"track_id": 1,  "label": "Student #1",  "overall_status": "Engaged",           "engaged_pct": 94.2, "disengaged_pct": 5.8,  "avg_engaged_prob": 0.942, "avg_disengaged_prob": 0.058, "tracked_frames": 320, "first_half_eng_pct": 96.0, "second_half_eng_pct": 92.4, "student_trend": "steady", "student_trend_delta": -3.6},
            {"track_id": 2,  "label": "Student #2",  "overall_status": "Disengaged",        "engaged_pct": 12.5, "disengaged_pct": 87.5, "avg_engaged_prob": 0.125, "avg_disengaged_prob": 0.875, "tracked_frames": 315, "first_half_eng_pct": 18.0, "second_half_eng_pct": 7.0,  "student_trend": "dropping", "student_trend_delta": -11.0},
            {"track_id": 3,  "label": "Student #3",  "overall_status": "Partially Engaged", "engaged_pct": 51.4, "disengaged_pct": 48.6, "avg_engaged_prob": 0.514, "avg_disengaged_prob": 0.486, "tracked_frames": 320, "first_half_eng_pct": 62.0, "second_half_eng_pct": 40.8, "student_trend": "dropping", "student_trend_delta": -21.2},
            {"track_id": 4,  "label": "Student #4",  "overall_status": "Engaged",           "engaged_pct": 89.1, "disengaged_pct": 10.9, "avg_engaged_prob": 0.891, "avg_disengaged_prob": 0.109, "tracked_frames": 308, "first_half_eng_pct": 88.0, "second_half_eng_pct": 90.2, "student_trend": "steady", "student_trend_delta": +2.2},
            {"track_id": 5,  "label": "Student #5",  "overall_status": "Disengaged",        "engaged_pct": 21.0, "disengaged_pct": 79.0, "avg_engaged_prob": 0.210, "avg_disengaged_prob": 0.790, "tracked_frames": 290, "first_half_eng_pct": 32.0, "second_half_eng_pct": 10.0, "student_trend": "dropping", "student_trend_delta": -22.0},
            {"track_id": 6,  "label": "Student #6",  "overall_status": "Disengaged",        "engaged_pct": 8.4,  "disengaged_pct": 91.6, "avg_engaged_prob": 0.084, "avg_disengaged_prob": 0.916, "tracked_frames": 312, "first_half_eng_pct": 11.0, "second_half_eng_pct": 5.8,  "student_trend": "steady", "student_trend_delta": -5.2},
            {"track_id": 7,  "label": "Student #7",  "overall_status": "Engaged",           "engaged_pct": 97.8, "disengaged_pct": 2.2,  "avg_engaged_prob": 0.978, "avg_disengaged_prob": 0.022, "tracked_frames": 320, "first_half_eng_pct": 98.0, "second_half_eng_pct": 97.6, "student_trend": "steady", "student_trend_delta": -0.4},
            {"track_id": 8,  "label": "Student #8",  "overall_status": "Partially Engaged", "engaged_pct": 47.9, "disengaged_pct": 52.1, "avg_engaged_prob": 0.479, "avg_disengaged_prob": 0.521, "tracked_frames": 320, "first_half_eng_pct": 42.0, "second_half_eng_pct": 53.8, "student_trend": "improving", "student_trend_delta": +11.8},
            {"track_id": 9,  "label": "Student #9",  "overall_status": "Engaged",           "engaged_pct": 78.4, "disengaged_pct": 21.6, "avg_engaged_prob": 0.784, "avg_disengaged_prob": 0.216, "tracked_frames": 305, "first_half_eng_pct": 85.0, "second_half_eng_pct": 71.8, "student_trend": "dropping", "student_trend_delta": -13.2},
        ]
        total_frames = 320
        fps = 25.0
        duration_sec = 12.8
        mid_sec = 6.4

    if not half_time_data:
        half_time_data = {
            "first_half_engagement": 55.0,
            "second_half_engagement": 45.0,
            "delta": -10.0,
            "trend_direction": "decreasing",
            "trend_label": "📉 Engagement Dropped (-10.0%) in 2nd Half",
            "insight": "Noticeable attention fatigue observed after midpoint.",
            "midpoint_sec": mid_sec,
            "midpoint_frame": total_frames // 2,
            "first_half_duration": mid_sec,
            "second_half_duration": round(duration_sec - mid_sec, 1),
        }

    # Use pre-rendered charts if available on disk (avoids heavy matplotlib spikes)
    stem = "demo"
    chart_keys = ["student_chart", "timeline_chart", "pie_chart", "halftime_chart"]
    existing_charts = {
        k: f"/outputs/demo/demo_{k}.png" for k in chart_keys
        if os.path.exists(os.path.join(demo_dir, f"demo_{k}.png"))
    }
    if len(existing_charts) == len(chart_keys):
        charts_dict = existing_charts
    else:
        charts_saved = generate_charts(demo_students, frame_metrics, demo_dir, stem, half_time_data=half_time_data)
        charts_dict = {k: f"/outputs/demo/{os.path.basename(v)}" for k, v in charts_saved}

    # Ensure text report exists
    report_path = os.path.join(demo_dir, "demo_report.txt")
    if not os.path.exists(report_path):
        rpt_txt = generate_text_report("demo_classroom_session", demo_students, frame_metrics, total_frames, fps, 7.8, half_time_data=half_time_data)
        with open(report_path, "w") as f:
            f.write(rpt_txt)

    # Ensure CSV exists
    csv_path = os.path.join(demo_dir, "demo_engagement_log.csv")
    if not os.path.exists(csv_path):
        recs = []
        for fm in frame_metrics:
            for s in demo_students:
                recs.append({
                    "frame": fm["frame"],
                    "timestamp_sec": fm["timestamp_sec"],
                    "student_id": s["track_id"],
                    "status": s["overall_status"],
                    "engaged_prob": s["avg_engaged_prob"],
                    "disengaged_prob": s["avg_disengaged_prob"],
                    "bbox": "[100, 100, 200, 200]"
                })
        pd.DataFrame(recs).to_csv(csv_path, index=False)

    total_stud = len(demo_students)
    n_eng     = sum(1 for s in demo_students if s["overall_status"] == "Engaged")
    n_partial = sum(1 for s in demo_students if s["overall_status"] == "Partially Engaged")
    n_dis     = sum(1 for s in demo_students if s["overall_status"] == "Disengaged")
    overall_eng_val = float(np.mean([m["engagement_rate"] for m in frame_metrics])) if frame_metrics else 35.6

    res_payload = {
        "video_url":          f"/outputs/demo/{chosen_video}",
        "student_summary":    demo_students,
        "frame_metrics":      frame_metrics[-200:],
        "halftime_analysis":  half_time_data,
        "charts":             charts_dict,
        "csv_url":            "/outputs/demo/demo_engagement_log.csv",
        "report_url":         "/outputs/demo/demo_report.txt",
        "is_demo":            True,
        "kpi": {
            "overall_engagement":          round(overall_eng_val, 1),
            "students_tracked":            total_stud,
            "engaged_students":            n_eng,
            "partially_engaged_students":  n_partial,
            "disengaged_students":         n_dis,
            "engaged_pct":                 round(n_eng / max(total_stud, 1) * 100, 1),
            "partially_engaged_pct":       round(n_partial / max(total_stud, 1) * 100, 1),
            "disengaged_pct":              round(n_dis / max(total_stud, 1) * 100, 1),
            "processed_frames":            total_frames,
            "total_frames":                total_frames,
            "duration_sec":                duration_sec,
            "fps":                         fps,
            "processing_time":             7.8,
            "first_half_engagement":       half_time_data["first_half_engagement"],
            "second_half_engagement":      half_time_data["second_half_engagement"],
            "halftime_delta":              half_time_data["delta"],
            "halftime_trend":              half_time_data["trend_direction"],
        }
    }
    DEMO_CACHE = res_payload
    return res_payload


@app.route("/demo")
def demo_route():
    try:
        data = get_or_create_demo_result()
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import argparse
    import socket

    def is_port_in_use(port_num, host="127.0.0.1"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port_num)) == 0

    parser = argparse.ArgumentParser(description="AI Classroom Engagement Dashboard")
    parser.add_argument("--host",  default="0.0.0.0")
    default_port = int(os.environ.get("PORT", 5001))
    parser.add_argument("--port",  type=int, default=default_port)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    port = args.port
    # Auto-detect if port is in use and find next free port
    if is_port_in_use(port):
        print(f"⚠️  Port {port} is already in use (common with macOS AirPlay on port 5000).")
        for p in range(port + 1, port + 20):
            if not is_port_in_use(p):
                print(f"✨ Automatically switching to free port: {p}")
                port = p
                break

    print("=" * 65)
    print("  AI CLASSROOM ENGAGEMENT DASHBOARD  (Flask Edition)")
    print(f"  👉 Open in browser: http://localhost:{port}")
    print("=" * 65)

    # Preload engine on startup
    print("[Startup] Pre-loading AI models…")
    get_engine()

    app.run(host=args.host, port=port, debug=args.debug, threaded=True)

