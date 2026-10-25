FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Install system dependencies (FFmpeg, OpenCV, OpenGL runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (lightweight CPU PyTorch for fast cloud build)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO face weights into root so container starts instantly
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt', 'yolov8n-face.pt')"

# Copy application files
COPY . /app

# Ensure user permissions for Hugging Face Spaces and non-root runtimes
RUN useradd -m -u 1000 user || true
RUN chown -R 1000:1000 /app
USER 1000

# Expose default port
EXPOSE 7860

CMD ["python", "dashboard/app.py", "--host", "0.0.0.0"]
