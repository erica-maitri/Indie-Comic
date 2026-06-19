FROM python:3.11-slim

# Set environment variable to run python unbuffered
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (mesa gl and git for huggingface/github downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install uv for rapid Python packaging
RUN pip install --no-cache-dir uv

# Copy requirements and install CPU-only torch first, then rest of deps
COPY indie_comic_pipeline/requirements.txt /app/indie_comic_pipeline/requirements.txt

# Install PyTorch CPU-only build to avoid CUDA download (~2 GB vs ~8 GB)
RUN uv pip install --system \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN uv pip install --system -r /app/indie_comic_pipeline/requirements.txt

# Copy project files
COPY . /app

# Grant execution rights to the entrypoint script
RUN chmod +x /app/entrypoint.sh

# Expose port 5000 (standard web UI port)
EXPOSE 5000

# Set entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
