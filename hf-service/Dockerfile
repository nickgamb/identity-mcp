# PyTorch 2.2.2 with CUDA 11.8 - supports Pascal GPUs (P40 = sm_61)
# Pre-built image = no downloading torch/CUDA during build
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/

# Install Python dependencies (torch already in base image)
# GPT-OSS requires transformers v4.55.0+ (released Aug 2025)
RUN pip install --no-cache-dir --timeout=300 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    transformers>=4.55.0 \
    accelerate>=0.30.0 \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    huggingface-hub>=0.23.0 \
    peft>=0.11.0

# Copy service files
COPY hf_api.py /app/

EXPOSE 8000

CMD ["python", "hf_api.py"]

