FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/

# Install Python dependencies with extended timeout for large packages (torch, cuda libs)
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy service files
COPY hf_api.py /app/

EXPOSE 8000

CMD ["python", "hf_api.py"]

