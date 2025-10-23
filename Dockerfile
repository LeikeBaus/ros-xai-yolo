# --- Basis: ROS Jazzy auf Ubuntu 24.04 ---
FROM ros:jazzy

# --- Update und Installation von Tools ---
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    && rm -rf /var/lib/apt/lists/*

# --- Python Pakete installieren ---
RUN pip install --no-cache-dir --break-system-packages --ignore-installed numpy \
    ultralytics \
    opencv-python \
    matplotlib \
    torch torchvision torchaudio \
    grad-cam


# --- Arbeitsverzeichnis festlegen ---
WORKDIR /workspace

# --- Standardkommando ---
CMD ["/bin/bash"]

