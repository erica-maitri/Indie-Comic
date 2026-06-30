FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

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

# Copy requirements
COPY indie_comic_pipeline/requirements.txt /app/indie_comic_pipeline/requirements.txt

# Install remaining requirements using pre-installed torch environment
RUN uv pip install --system -r /app/indie_comic_pipeline/requirements.txt

# Copy project files
COPY . /app

# Grant execution rights to the entrypoint script
RUN chmod +x /app/entrypoint.sh

# Expose port 5000 (standard web UI port)
EXPOSE 5000

# Set entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
