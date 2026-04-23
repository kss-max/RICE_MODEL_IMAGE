# ============================================
# Dockerfile for Rice Blast Detection API
# ============================================
# 
# WHAT IS A DOCKERFILE?
# It's a recipe that tells Docker how to build a "container" —
# think of it as a lightweight virtual machine with its own
# Python, TensorFlow, and your app code inside.
#
# WHY python:3.10-slim?
# - Python 3.10 supports TensorFlow 2.19.0 (same as Colab)
# - "slim" = minimal OS without extra tools = smaller image size
# ============================================

FROM python:3.10-slim

# ── Set working directory inside the container ──
# All commands after this run inside /app
WORKDIR /app

# ── Install OS-level dependencies ──
# TensorFlow needs some system libraries to work
# We clean up after to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first (Docker cache optimization) ──
# WHY COPY THIS SEPARATELY?
# Docker caches each step. If you change app.py but NOT requirements.txt,
# Docker will skip the pip install step (uses cache) = faster rebuilds!
COPY requirements.txt .

# ── Install Python dependencies ──
# --no-cache-dir = don't store pip's cache (saves ~100MB in image)
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy the rest of the app ──
COPY app.py .

# ── Copy model file (for local testing) ──
# We are commenting this out for Render deployment!
# The app will automatically download it from Google Drive on startup.
# COPY paddy_disease_classification_model.keras .

# ── Expose port 8000 ──
# This tells Docker "this container listens on port 8000"
# It doesn't actually open the port — that's done with -p flag when running
EXPOSE 8000

# ── Start the server ──
# This command runs when the container starts
# --host 0.0.0.0 = accept connections from outside the container
# --port 8000 = listen on port 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
