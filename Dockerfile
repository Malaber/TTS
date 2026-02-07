FROM python:3.10-slim

# Install system dependencies + curl to get Rust
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc g++ make python3-dev espeak-ng libsndfile1-dev ffmpeg \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust compiler (required for sudachipy on ARM64)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch
RUN pip install --no-cache-dir torch torchaudio

WORKDIR /app
COPY . /app

# This should now succeed in compiling sudachipy
RUN make install

ENTRYPOINT ["tts"]
CMD ["--help"]