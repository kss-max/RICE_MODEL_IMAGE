# ============================================
# Dockerfile for Hugging Face Spaces
# ============================================
FROM python:3.10-slim

# ── Hugging Face Security Requirement ──
# HF Spaces run Docker containers as a non-root user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# ── Install OS Dependencies ──
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER user

# ── Python Dependencies ──
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Copy Application & Model ──
COPY --chown=user . .

# ── Hugging Face Default Port ──
EXPOSE 7860

# ── Start Server ──
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
