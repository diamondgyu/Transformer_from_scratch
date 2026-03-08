# syntax=docker/dockerfile:1

FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    WANDB_DIR=/tmp/wandb

WORKDIR /opt/ml/code

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# CPU by default (good for SageMaker Serverless).
# For EC2 GPU build, pass:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN pip install torch torchvision torchaudio --index-url ${TORCH_INDEX_URL}

RUN pip install \
    transformers \
    datasets \
    pandas \
    openpyxl \
    tqdm \
    wandb \
    sacrebleu \
    sentencepiece

COPY src ./src
COPY README.md ./README.md

ENV PYTHONPATH=/opt/ml/code

# Default behavior: start training script.
CMD ["python", "src/main.py"]
