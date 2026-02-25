FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for uv and MuJoCo
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgl1-mesa-dev \
    libosmesa6-dev \
    patchelf \
    python3 \
    python3-pip \
    python3-venv \
    unzip \
    wget \
    vim \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://starship.rs/install.sh | sh -s -- --yes

# The 'ubuntu' user (UID 1000) is standard in Ubuntu 24.04 images.
# Ensure it is configured with a bash shell and home directory.
RUN usermod -s /bin/bash ubuntu && \
    mkdir -p /home/ubuntu && \
    chown ubuntu:ubuntu /home/ubuntu

# Set environment variables for MuJoCo
# MUJOCO_GL=egl is essential for hardware-accelerated rendering in Docker with NVIDIA GPUs.
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Switch to ubuntu user
USER ubuntu
WORKDIR /home/ubuntu

# Install ubuntu's local dependencies
RUN mkdir -p /home/ubuntu/.mujoco \
 && curl -L https://github.com/google-deepmind/mujoco/releases/download/3.5.0/mujoco-3.5.0-linux-x86_64.tar.gz | tar -xz -C /home/ubuntu/.mujoco \
 && curl -LsSf https://astral.sh/uv/install.sh | sh \
 && /home/ubuntu/.local/bin/uv python install 3.14 \
 &&  echo 'eval "$(starship init bash)"' >> ~/.bashrc

# Set environment variables for installing/using mujoco with python
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco-3.2.7/bin
ENV MUJOCO_PATH=/home/ubuntu/.mujoco/mujoco-3.2.7

# Set up environment path and default python
ENV PATH="/home/ubuntu/.local/bin:$PATH"
ENV UV_PYTHON=3.14

# Default command
CMD ["bash"]

# docker run --rm --gpus all -it turn-taking:latest 
