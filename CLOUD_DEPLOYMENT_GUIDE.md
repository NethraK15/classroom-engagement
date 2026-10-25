# 🌐 Cloud & Public Deployment Guide
## Deploying Real-Time Classroom Engagement AI for 24/7 Public Access

---

## ⚡ 1. Instant Live Public Link (Active Right Now)

Your local Flask server is currently exposed to the internet via an encrypted HTTPS tunnel:

> 🔗 **Live Public URL**: **`https://41c8304fb3d5ce.lhr.life`**
> 
> *(Try opening this URL right now on your phone or sending it to a friend!)*

### What works over this live link:
- ✅ **Webcam Recording**: Because the URL uses **`https://`**, modern mobile and desktop browsers (Chrome, Safari, Edge) allow camera access.
- ✅ **Demo Session**: Clicking **"Try Demo Session"** instantly displays the 362-frame annotated video, 24 tracked students, halftime engagement shift (+2.8%), and all 4 analytical charts.
- ✅ **Video Upload & Analysis**: Uploading videos runs through the YOLOv8 + ByteTrack + EfficientNet-B0 + Binary Transformer pipeline.

> ⚠️ **Note about Tunnels**: This instant link remains online as long as your machine is running and connected to WiFi. If you close your laptop, the tunnel disconnects. For a **permanent link that stays live 24/7**, use **Option 2** below.

---

## ☁️ 2. Permanent 24/7 Cloud Hosting — Hugging Face Spaces (100% Free with Gradio SDK)

> 💡 **Why Docker showed as paid**: Hugging Face recently moved raw Docker containers to a paid tier ($9/mo PRO).
> **Static** is free, but Static can **only** serve plain HTML/CSS (it cannot run Python, PyTorch, or your AI models).
> The **Gradio SDK** is **100% FREE** with **16 GB RAM + 2 vCPU**, no credit card required, and supports our custom launcher!

### Why Gradio SDK is the best free option:
- **100% Free**: Includes **16 GB RAM** and **2 vCPU** (runs PyTorch + YOLOv8 + Transformer).
- **Always Online (24/7)**: Stays online permanently without needing your computer.
- **Permanent URL**: You get a permanent public link:
  `https://huggingface.co/spaces/<your-username>/classroom-engagement`
- **Automatic SSL/HTTPS**: Webcam `getUserMedia()` works for all visitors globally.

### Step-by-Step Instructions:

#### Step A: Create a Free Account
1. Go to [huggingface.co/join](https://huggingface.co/join) (if not already logged in).

#### Step B: Create a New Space
1. Click your profile picture (top right) → **New Space**.
2. Set **Space name**: `classroom-engagement`.
3. Select **Space SDK**: Choose **Gradio** (this is the **Free** option with 16 GB RAM!).
4. Space hardware: Keep default **CPU basic • 2 vCPU • 16GB • Free**.
5. Set visibility: **Public**.
6. Click **Create Space**.

#### Step C: Upload / Push the Code
In your terminal:

```bash
# 1. In your project directory:
cd /Users/nethra/Downloads/colab_binary_transformer

# 2. Add files and commit:
git init
git add .
git commit -m "Deploy Classroom Engagement Dashboard with Gradio SDK"

# 3. Add Hugging Face Space as a remote and push (replace YOUR_USERNAME):
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/classroom-engagement
git push --force space main
```

*(Alternatively, you can upload the files directly using the Hugging Face web interface: click **Files** → **Add file** → **Upload files**)*.

#### Step D: Automatic Build & Launch
- Hugging Face automatically detects [`app.py`](file:///Users/nethra/Downloads/colab_binary_transformer/app.py) and [`requirements.txt`](file:///Users/nethra/Downloads/colab_binary_transformer/requirements.txt), installs dependencies, and boots the dashboard on port `7860`.
- Within 2–3 minutes, your Space status will turn **"Running"**.
- Anyone with your Space URL (`https://huggingface.co/spaces/YOUR_USERNAME/classroom-engagement`) can now visit, record webcam, view the demo, and analyze videos 24/7!

---

## 🚀 3. Alternative: Render.com (Free Web Service)

If you prefer Render:
1. Create a free account at [render.com](https://render.com).
2. Push your project to a GitHub repository.
3. In Render Dashboard, click **New +** → **Web Service**.
4. Connect your GitHub repository.
5. Select **Environment**: **Docker**.
6. Render will automatically build the `Dockerfile` and assign an HTTPS URL like `https://classroom-engagement.onrender.com`.

---

## ⚡ 3. Alternative: Google Colab + Free GPU (Fastest Video Inference)

If you want free **GPU acceleration** (NVIDIA T4 16GB) so video analysis finishes in 2–5 seconds instead of 30–60 seconds on CPU:

1. Open your Google Drive, go to your existing `colab_binary_transformer` folder, and open any Colab notebook.
2. In a new notebook cell, run:
```python
# Cell 1: Mount Drive & install tunnel
from google.colab import drive
drive.mount('/content/drive')
!pip install -q pyngrok flask ultralytics lapx

# Cell 2: Connect ngrok (free auth token from https://ngrok.com)
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
public_url = ngrok.connect(5001)
print("🚀 Live Public Dashboard URL:", public_url)

# Cell 3: Start dashboard server
%cd /content/drive/MyDrive/.../colab_binary_transformer/dashboard
!python app.py --port 5001
```
Anyone opening that `public_url` will connect to your Colab-powered dashboard with full GPU inference!

---

## 🔒 4. Re-launching the Instant Tunnel at Any Time

Whenever you are on your laptop (Mac or Windows) and want to spin up a quick public HTTPS link:

### On macOS / Linux:
```bash
ssh -o StrictHostKeyChecking=no -R 80:localhost:5001 nokey@localhost.run
```

### On Windows:
Open PowerShell or Command Prompt and run:
```cmd
ssh -R 80:localhost:5001 nokey@localhost.run
```
*(OpenSSH is installed by default on Windows 10/11)*. It will output an HTTPS URL instantly!
