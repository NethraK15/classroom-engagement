# 🚀 EngageAI Dashboard — Complete Deployment Guide

> **System**: Binary Engagement Transformer (Engaged vs Disengaged)  
> **Pipeline**: YOLOv8 + ByteTrack → EfficientNet-B0 (1280) + OpenFace (709) → 1989-dim → Transformer  
> **Backend**: Python Flask  
> **Frontend**: Vanilla HTML/CSS/JS (no Gradio, no Node.js required)

---

## 📁 File Structure

```
colab_binary_transformer/
├── models_binary_transformer/
│   └── binary_transformer_weights.pth  ← trained model (already here!)
├── config_binary_transformer.py
├── models_binary_transformer.py
├── feature_extractor_binary.py
├── dashboard/                          ← NEW dashboard folder
│   ├── app.py                          ← Flask server (entry point)
│   ├── requirements.txt
│   ├── DEPLOY_GUIDE.md                 ← this file
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/main.js
│   └── outputs/                        ← auto-created, stores job results
```

---

## ✅ Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python      | 3.9+    | 3.10 or 3.11 recommended |
| pip         | latest  | `pip install --upgrade pip` |
| ffmpeg      | any     | For browser-compatible video re-encoding |
| RAM         | 4 GB+   | 8 GB recommended for multi-student videos |
| GPU         | optional| CUDA GPU speeds up EfficientNet & Transformer |

---

## 🖥️ Option A — Local Machine (Fastest to Try)

### Step 1 — Navigate to dashboard folder

```bash
cd /path/to/colab_binary_transformer/dashboard
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users (CUDA 11.8):** Install PyTorch with CUDA support first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```

### Step 4 — Install ffmpeg (for video re-encoding)

| OS | Command |
|----|---------|
| macOS | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| Windows | Download from https://ffmpeg.org/download.html and add to PATH |

### Step 5 — Run the dashboard

```bash
python app.py
```

Expected output:
```
=================================================================
  AI CLASSROOM ENGAGEMENT DASHBOARD  (Flask Edition)
  http://localhost:5000
=================================================================
[Startup] Pre-loading AI models…
[Engine] Device: cpu         ← or cuda if GPU available
[Engine] Loading checkpoint: ../models_binary_transformer/binary_transformer_weights.pth
[Engine] Ready.
 * Running on http://0.0.0.0:5000
```

### Step 6 — Open in browser

Navigate to **http://localhost:5000** 

---

## 🔧 Configuration Options

```bash
# Run on a custom port
python app.py --port 8080

# Run with debug mode (auto-reload, verbose errors)
python app.py --debug

# Bind to specific interface
python app.py --host 127.0.0.1 --port 5000
```

---

## ☁️ Option B — Google Colab Deployment (Public Link)

### Step 1 — Mount Drive and navigate

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys
proj = "/content/drive/MyDrive/Student_Engagement_DL/colab_binary_transformer"
sys.path.insert(0, proj)
os.chdir(proj)
```

### Step 2 — Install dependencies

```bash
!pip install -q flask pyngrok ultralytics lapx opencv-python-headless torch torchvision
```

### Step 3 — Launch with ngrok tunnel

```python
from pyngrok import ngrok
import subprocess, threading

def run_flask():
    subprocess.run(["python", "dashboard/app.py", "--port", "5000"])

threading.Thread(target=run_flask, daemon=True).start()

import time; time.sleep(6)  # wait for Flask to start

# Create public tunnel
public_url = ngrok.connect(5000)
print(f"\n  Dashboard live at: {public_url}\n")
```

> **Note:** ngrok free tier requires signing up at https://ngrok.com and setting your authtoken:
> ```python
> from pyngrok import conf
> conf.get_default().auth_token = "YOUR_NGROK_TOKEN"
> ```

---

## 🖥️ Option C — Deploy on a Linux Server / VPS

### Step 1 — Install system dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg git
```

### Step 2 — Clone and install

```bash
git clone <your-repo-url>
cd colab_binary_transformer/dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
```

### Step 3 — Run with Gunicorn (production WSGI)

```bash
# Use 1 worker only — processing is CPU/GPU bound
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 600 app:app
```

### Step 4 — Nginx reverse proxy (optional)

```nginx
server {
    listen 80;
    server_name your-domain.com;
    client_max_body_size 2G;
    proxy_read_timeout 600s;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}
```

---

## 🐳 Option D — Docker

### Dockerfile (in `dashboard/`):

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY .. /app

WORKDIR /app/dashboard
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]
```

### Build and run:

```bash
cd colab_binary_transformer
docker build -f dashboard/Dockerfile -t engageai .
docker run -p 5000:5000 engageai
```

---

## 📋 Pipeline Walkthrough

```
User uploads video
    |
    v
YOLOv8 (yolov8n-face.pt)
  - Face-specific detector
  - classes=[0] (person only when using generic model)
  - Conf threshold: 0.35
    |
    v
ByteTrack multi-face tracking
  - Persistent track IDs across frames
  - Eliminates ID switches / ghost splits
  - Min track frames filter removes false detections
    |
    v
Haar cascade face verification
  - Validates actual human face inside bounding box
  - Rejects bags, chairs, laptops, table corners
    |
    v
Face crop extracted per student
    |
    +——————————————————————————————+
    |                              |
    v                              v
EfficientNet-B0              OpenFace Dynamics
  1280-dim visual                709-dim behavioral
  features from                  (head pose, eye gaze,
  face crop                      action units, landmarks)
    |                              |
    +——————————————————————————————+
                   |
                   v
         Fused 1989-dim vector
         (normalized with training stats)
                   |
                   v
         Binary Engagement Transformer
           d_model=128, 3 layers, 4 heads
           120-frame sliding window buffer
                   |
                   v
         Softmax → [P(engaged), P(disengaged)]
                   |
          +-———————+———————+
          |                 |
     Engaged           Disengaged
    (Green box)        (Red box)
```

---

## 🎯 Output Descriptions

| Output | Description |
|--------|-------------|
| **Annotated Video** | MP4 with colored bounding boxes. Green = Engaged, Red = Disengaged. Each face shows `#ID E:XX% D:XX%` probability overlay |
| **Per-Student Cards** | 2x1 probability matrix (Engaged score / Disengaged score), bars, borderline detection |
| **Student Chart** | Side-by-side bar chart of avg engaged/disengaged probability per student |
| **Timeline Chart** | Smoothed classroom engagement % over video duration |
| **Pie Chart** | Donut chart of overall Engaged vs Disengaged class split |
| **CSV Log** | Per-frame: frame#, timestamp, student_id, status, engaged_prob, disengaged_prob, bbox |
| **Text Report** | Classroom summary + full per-student analysis |

---

## 🎛️ Settings Explained

| Setting | Default | Effect |
|---------|---------|--------|
| **YOLO Confidence** | 0.35 | Min face detection confidence. Lower = more detections. Higher = stricter. |
| **Engagement Threshold** | 0.50 | P(engaged) >= threshold → Engaged. Raise to 0.60 if too many false Engaged labels. |
| **Temporal Buffer** | 120 | Rolling frame buffer per student. 120 matches training window exactly. |
| **Min Track Frames** | 12 | Ignores students tracked fewer than N frames (filters ghost detections). |
| **CLAHE Enhancement** | Off | Applies CLAHE + sharpening. Helps in dim classrooms. Adds ~20% processing time. |

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics lapx` |
| `No module named 'config_binary_transformer'` | Run `app.py` from inside the `dashboard/` directory |
| Too many ghost students detected | Increase Min Track Frames to 20-30 |
| Students detected as objects/bags | Use `yolov8n-face.pt` (auto-downloaded on first run) |
| Video won't play in browser | Install ffmpeg — the re-encoding step requires it |
| Out of memory | Reduce Temporal Buffer to 32-64 frames |
| GPU not being used | Install CUDA PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |

---

*EngageAI Dashboard — Flask Edition — No Gradio Required*
