"""
app.py — Root Cloud & Hugging Face Spaces Launcher
─────────────────────────────────────────────────────────────────────────────
Serves the AI Classroom Engagement Dashboard on cloud platforms (Hugging Face
Spaces with Gradio SDK, Render, Railway, or VPS).
─────────────────────────────────────────────────────────────────────────────
"""
import os
import sys

# Add root and dashboard to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(SCRIPT_DIR, "dashboard")
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dashboard.app import app as flask_app

# Default app object for WSGI / ASGI runners
app = flask_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[*] Launching AI Classroom Engagement Dashboard on port {port}...")
    try:
        from a2wsgi import WSGIMiddleware
        import uvicorn
        asgi_app = WSGIMiddleware(flask_app)
        uvicorn.run(asgi_app, host="0.0.0.0", port=port)
    except Exception:
        flask_app.run(host="0.0.0.0", port=port)
