#Use Python base image
FROM python:3.10-slim

# environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# system dependencies for OpenCV, Matplotlib, and Qt support
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-xcb1 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libqt5gui5 \
    libqt5opengl5 \
    libqt5widgets5 \
    libqt5network5 \
    libqt5core5a \
    libfreetype6-dev \
    pkg-config \
    git \
    curl && \
    apt-get clean

#Python dependencies 
RUN pip install --upgrade pip && \
    pip install torch torchvision numpy==1.23.5 scipy==1.9.3 \
    opencv-python pandas timm Pillow matplotlib

# Apache Beam and Google Cloud dependencies
RUN pip install apache-beam[gcp] google-cloud-storage google-cloud-pubsub


COPY . /app
WORKDIR /app

#Download YOLOv5 and MiDaS models to avoid re-downloading
RUN python -c "import torch; \
               torch.hub.load('ultralytics/yolov5', 'yolov5s'); \
               torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')"

#Set the entry point to run the script
ENTRYPOINT ["python3", "DataFlow.py"]
