FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000
WORKDIR /app

COPY agent/requirements.txt .
RUN pip install -r requirements.txt

COPY agent .
# default command – override in Coolify if you still need “dev”
CMD ["python","main.py","dev"]
EXPOSE 8000
