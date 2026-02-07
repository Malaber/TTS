# Use a native ARM64 Python base instead of NVIDIA/CUDA
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc g++ make python3-dev espeak-ng libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Fix the VersionConflict error by upgrading build tools immediately
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch (CPU version for Mac architecture)
RUN pip install --no-cache-dir torch torchaudio

# Copy TTS repository contents
WORKDIR /app
COPY . /app

# Install TTS
RUN make install

ENTRYPOINT ["tts"]
CMD ["--help"]