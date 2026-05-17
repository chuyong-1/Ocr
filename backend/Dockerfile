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

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY backend/ ./backend/
COPY models/  ./models/

# Runtime directories (volumes will overlay these in compose)
RUN mkdir -p data/uploads data/results

EXPOSE 8000
