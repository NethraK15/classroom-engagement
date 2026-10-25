# 🪟 Complete Windows Deployment Guide
## AI Classroom Engagement Detection System
**YOLOv8 + ByteTrack + EfficientNet-B0 (1280) + OpenFace C++ (709) → 1989-dim → Binary Transformer**

---

## 🎯 Architecture & Verified Features
This system is configured to run all 5 core capabilities seamlessly on **Windows 10/11**:

| Feature | Windows Implementation | Verification |
| :--- | :--- | :--- |
| 📹 **Live Webcam** | HTML5 `getUserMedia` + `MediaRecorder` in browser | Record & submit live video directly to inference pipeline |
| 👤 **OpenFace C++** | Official 64-bit `FeatureExtraction.exe` standalone binary | 709-dim pose, gaze, landmarks & Action Units (AUs) |
| 🧠 **EfficientNet-B0** | PyTorch torchvision deep visual embedding model | 1280-dim feature vector per student face crop |
| ⚡ **Binary Transformer** | Custom temporal sequence encoder (`d_model=128`, 3 layers) | Fused 1989-dim vector → Softmax Engaged vs Disengaged |
| 🎬 **Demo Session** | Instant pre-computed session with H.264 annotated video | 24 tracked students, halftime shift, and 4 charts |

---

## 📦 Required Downloads for Windows

Before starting, download these 3 packages on your Windows laptop:

### 1. Python 3.10 or 3.11 (64-bit)
- Download from [python.org](https://www.python.org/downloads/windows/).
- ⚠️ **Important during installation**: Check the box **"Add Python to PATH"** on the very first screen.

### 2. OpenFace 2.2.0 Standalone Binaries (Win x64)
- Download official release: [OpenFace_2.2.0_win_x64.zip (GitHub Release)](https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip) (~250 MB).
- Extract the zip folder to:
  `C:\OpenFace_2.2.0_win_x64`  *(or directly inside your project folder)*.

### 3. FFmpeg for Windows (For video rendering)
- Download `ffmpeg-release-essentials.zip` from [gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/).
- Extract `ffmpeg.exe` to `C:\ffmpeg\bin` and add `C:\ffmpeg\bin` to your Windows System `PATH` (or place `ffmpeg.exe` directly in `colab_binary_transformer\dashboard\`).

### 4. Microsoft Visual C++ Redistributable (if not already installed)
- [Visual C++ 2015–2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)

---

## 🚀 Step-by-Step Deployment Instructions

### Step 1: Transfer Project to Windows Laptop
Copy the entire `colab_binary_transformer` project directory to your Windows machine (for example, to `C:\colab_binary_transformer` or `D:\colab_binary_transformer`).

Make sure the model weight file exists:
- `colab_binary_transformer\models_binary_transformer\binary_transformer_weights.pth` (7.6 MB)

---

### Step 2: Set Up OpenFace Models
OpenFace requires its facial landmark CEN patch expert models.
1. Open Windows **PowerShell** as Administrator or standard user.
2. Navigate to your extracted OpenFace folder:
   ```powershell
   cd C:\OpenFace_2.2.0_win_x64
   ```
3. Run the model download script:
   ```powershell
   # If PowerShell blocks script execution, run:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\download_models.ps1
   ```
   *(This downloads the missing `.dat` patch files into `model\patch_experts\`)*.
4. Test that OpenFace is working:
   ```powershell
   .\FeatureExtraction.exe -help
   ```
   If it prints the OpenFace help text, OpenFace is 100% ready!

---

### Step 3: Create Python Virtual Environment
Open **Command Prompt (CMD)** or **PowerShell**:

```cmd
cd C:\colab_binary_transformer\dashboard
python -m venv venv
venv\Scripts\activate
```

*(You will see `(venv)` appear at the start of your command prompt)*.

---

### Step 4: Install Dependencies

#### A. Install PyTorch:
- **If you have an NVIDIA GPU (GeForce / RTX / GTX):**
  ```cmd
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- **If you have an Intel or AMD CPU only:**
  ```cmd
  pip install torch torchvision
  ```

#### B. Install Remaining Dependencies:
```cmd
pip install -r requirements.txt
```
*(This installs Flask, OpenCV, Ultralytics YOLOv8, ByteTrack lapx, pandas, matplotlib, tqdm, and Werkzeug)*.

---

### Step 5: Connect OpenFace to the Python Pipeline
The pipeline now automatically detects both unnested and nested extraction paths:
- `C:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe` *(default when unzipping)*
- `C:\OpenFace_2.2.0_win_x64\FeatureExtraction.exe`
- Or any subfolder inside `C:\OpenFace_2.2.0_win_x64\` (auto-detected via recursive scan)

To explicitly specify the path if needed:
```cmd
:: In Command Prompt:
set OPENFACE_EXE=C:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe

:: Or in PowerShell:
$env:OPENFACE_EXE="C:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
```

Verify Python detects OpenFace by running:
```cmd
python -c "from feature_extractor_binary import OpenFaceFeatureExtractor; fe = OpenFaceFeatureExtractor(); print('OpenFace Exe:', fe.openface_exe)"
```
Output will show:
`[OpenFace] C++ binary detected: C:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe`

---

### Step 6: Start the Dashboard Server
Inside `C:\colab_binary_transformer\dashboard`:

```cmd
python app.py --port 5001
```

Console will display:
```text
=================================================================
  AI CLASSROOM ENGAGEMENT DASHBOARD  (Flask Edition)
  👉 Open in browser: http://localhost:5001
=================================================================
[Startup] Pre-loading AI models…
[Engine] Device: cuda:0 (or cpu)
[Engine] Loading checkpoint: models_binary_transformer/binary_transformer_weights.pth
[OpenFace] C++ binary detected: C:\OpenFace_2.2.0_win_x64\FeatureExtraction.exe
[Engine] Ready.
 * Running on http://127.0.0.1:5001
```

---

## 🌐 Using All Features in Browser

Open **Google Chrome**, **Microsoft Edge**, or **Firefox** and visit:
```
http://localhost:5001
```

### 1. 📹 Webcam Recording & Analysis
1. In the dashboard top bar, click the **"Record Live Webcam"** tab.
2. Click **"Enable Webcam"** and allow camera permission in your browser.
3. Align student face(s) facing the camera.
4. Click **"Start Recording"** (record between 5 to 20 seconds).
5. Click **"Stop Recording"**. Preview the video.
6. Click **"Analyse Recorded Video"**.
7. The system streams real-time progress through YOLOv8 → ByteTrack → EfficientNet-B0 → OpenFace → Binary Transformer and displays the final annotated video and student report.

### 2. 🎬 Demo Part ("Try Demo Session")
1. Click the purple **"Try Demo Session"** button on the top right.
2. Instantly loads:
   - Full 362-frame annotated video with tracking boxes and HUD telemetry.
   - Classroom KPIs (24 students tracked, 35.7% engagement).
   - Half-time analysis breakdown (1st half 34.3% vs 2nd half 37.1%, +2.8% steady trend).
   - 4 analytical charts: Timeline, Student Distribution, Donut Split, and Halftime Delta.
   - Text report and CSV download links.

### 3. 📁 Video Upload
1. Switch to **"Upload Video File"** tab.
2. Drag and drop any classroom MP4/AVI video.
3. Configure detection confidence and thresholds if needed.
4. Click **"Run Engagement Analysis"**.

---

## 🛠️ Windows Troubleshooting & FAQs

### Q1: "Execution of scripts is disabled on this system" when running `download_models.ps1`
**Fix**: Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then re-run `.\download_models.ps1`.

### Q2: Port 5000 or 5001 is already occupied
**Fix**: The app automatically detects port conflicts and switches to the next free port (e.g., `5002`). You can also specify any port manually:
```cmd
python app.py --port 8080
```

### Q3: Webcam won't start in Chrome / Edge
**Fix**: Modern browsers require either `localhost`, `127.0.0.1`, or `https://` to access camera devices.
- Always access via `http://localhost:5001` or `http://127.0.0.1:5001`.
- Check Windows Settings: **Privacy & Security → Camera → Allow desktop apps to access your camera**.

### Q4: Is CUDA required?
**Fix**: No. If your Windows laptop does not have an NVIDIA GPU, PyTorch will automatically use the CPU. With CPU inference, video processing still runs smoothly with EfficientNet-B0 and YOLOv8n.
