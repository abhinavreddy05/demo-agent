# ─────────────────── agent/Dockerfile ───────────────────
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app
COPY agent/requirements.txt .
RUN pip install -r requirements.txt

# copy source
COPY agent .

# 🟢 pull the model weights once, at build‑time
RUN python main.py download-files      \
    && echo "Turn‑detector weights downloaded."

EXPOSE 8000
CMD ["python", "main.py", "dev"]
