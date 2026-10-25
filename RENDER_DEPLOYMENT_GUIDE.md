# 🚀 Complete Render.com Deployment Guide (100% Free)
## AI Classroom Engagement Detection System
**Get a permanent 24/7 public link with HTTPS, Webcam, and Demo features**

---

## 🌟 Why Render.com?
- **100% Free**: No credit card required.
- **Permanent Link**: Stays online 24/7 at `https://your-app-name.onrender.com`.
- **Automatic HTTPS/SSL**: Secure certificates are generated automatically, so browser **Webcam (`getUserMedia`) works out of the box** for anyone globally.
- **Automatic Redeploy**: Whenever you update your code and push to GitHub, Render automatically redeploys.

---

## 📋 Step 1: Create a Repository on GitHub

1. Open your browser and go to: **[github.com/new](https://github.com/new)** (Log in or create a free account if you haven't).
2. Enter **Repository name**: `classroom-engagement`
3. Set visibility to **Public** (or Private).
4. **Do NOT** check "Add a README file" or "Add .gitignore" (we already created them locally).
5. Click **Create repository**.

---

## 💻 Step 2: Push Local Code to GitHub

Your code is already committed locally. Run these commands in your terminal to push it to your GitHub repository:

```bash
cd /Users/nethra/Downloads/colab_binary_transformer

# Link to your new GitHub repo (Replace YOUR_USERNAME with your GitHub username):
git remote add origin https://github.com/YOUR_USERNAME/classroom-engagement.git
git branch -M main
git push -u origin main
```

*(If prompted, enter your GitHub username and Personal Access Token or sign in via browser)*.

---

## 🌐 Step 3: Connect to Render.com

1. Go to **[render.com](https://render.com)**.
2. Click **Sign In** (top right) → choose **"Sign in with GitHub"**.  
   *(This gives Render permission to read your repositories)*.
3. In the Render Dashboard, click the blue **"New +"** button (top right) and select **"Web Service"**.
4. Select **"Build and deploy from a Git repository"** and click **Next**.
5. Find your repository `classroom-engagement` from the list and click **Connect**.  
   *(If you don't see it, click "Configure access" to grant Render access to your GitHub repo)*.

---

## ⚙️ Step 4: Configure the Web Service Settings

Fill in the settings form as follows:

| Setting | Value to Enter |
| :--- | :--- |
| **Name** | `classroom-engagement` *(determines your URL)* |
| **Region** | Choose closest to you (e.g. *Singapore* or *Frankfurt*) |
| **Branch** | `main` |
| **Root Directory** | *(Leave blank)* |
| **Runtime** | **Python 3** |
| **Build Command** | `pip install --upgrade pip && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r dashboard/requirements.txt` |
| **Start Command** | `python dashboard/app.py` |
| **Instance Type** | Select **Free ($0 / month)** |

### Optional Environment Variable:
Scroll down to **Advanced** → **Add Environment Variable**:
- **Key**: `PYTHON_VERSION`
- **Value**: `3.11.9`

Click the blue **"Deploy Web Service"** button at the bottom!

---

## ⏳ Step 5: Watch the Build & Go Live!

1. Render will show the live build console.
2. It will install PyTorch (CPU-optimized, fast and lightweight), Ultralytics YOLOv8, Flask, and dependencies.
3. Within 3–4 minutes, you will see:
   ```text
   ==> Uploading build...
   ==> Build successful 🎉
   ==> Deploying...
   ==> Starting service with 'python dashboard/app.py'
   =================================================================
     AI CLASSROOM ENGAGEMENT DASHBOARD  (Flask Edition)
     👉 Open in browser: http://0.0.0.0:10000
   =================================================================
   ==> Your service is live 🎉
   ```
4. Look at the top left under your service name — you will see your live public link:
   👉 **`https://classroom-engagement.onrender.com`**

---

## 🎯 How Your App Functions on Render

- 📹 **Live Webcam**: Visitors click "Record Live Webcam" → Browser prompts for camera permissions (works because Render uses HTTPS) → Records video → Submits to AI pipeline.
- 🎬 **Demo Session**: Clicking "Try Demo Session" immediately plays the 362-frame annotated video, displays 24 tracked students, and renders all 4 analytical charts without running heavy models.
- 📁 **Video Upload**: Visitors can upload classroom MP4 files for processing.

> 💡 **Tip about Render Free Tier**: If no one visits your site for 15 minutes, Render puts the app in sleep mode to save energy. When someone clicks the link again, it automatically wakes up in ~30–45 seconds!
