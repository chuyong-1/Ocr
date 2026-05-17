FROM python:3.11-slim

# System deps: OpenCV, FFmpeg, fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1 libgomp1 \
    ffmpeg \
    fonts-dejavu-core \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create persistent directories
RUN mkdir -p uploads results

EXPOSE 8000
