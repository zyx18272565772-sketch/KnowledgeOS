FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    HF_ENDPOINT=https://hf-mirror.com \
    DEBIAN_FRONTEND=noninteractive

# Tesseract OCR（含简体中文包）以及 Pillow 等图像库的运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装 CPU 版 torch，避免默认 CUDA wheel 把镜像撑到数 GB
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY python-service/requirements.txt /app/python-service/requirements.txt
RUN pip install -r /app/python-service/requirements.txt

# 构建期预下载本地 Embedding 模型，容器启动无需再访问外网
ARG LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${LOCAL_EMBEDDING_MODEL}')"

# 保持与本地一致的目录结构：/app/python-service 与 /app/static 同级
COPY python-service/ /app/python-service/
COPY static/ /app/static/

WORKDIR /app/python-service

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]