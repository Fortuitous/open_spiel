# Use a PyTorch CUDA runtime with Python >= 3.11 for OpenSpiel compatibility
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# setup.py defaults to clang++
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    cmake \
    git \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install core Python dependencies (Cached layer)
# Note: Torch is already included in the base image
RUN pip install --no-cache-dir google-cloud-storage wandb

# Install OpenSpiel from local source
# We do this FIRST to cache the heavy C++ compilation
COPY . .
RUN pip install .

# Copy the specific trainer scripts LAST
# This ensures that any changes to these files result in a near-instant rebuild
COPY open_spiel/python/games/backgammon/trainer_v1.py /app/trainer_v1.py
COPY open_spiel/python/games/backgammon/expert_eyes_model.py /app/expert_eyes_model.py

# Patch the local paths in the scripts
RUN sed -i '/sys.path.append("\/home\/jeremy\/projects\/open_spiel")/d' /app/trainer_v1.py && \
    sed -i '/sys.path.append("\/home\/jeremy\/projects\/open_spiel\/build\/python")/d' /app/trainer_v1.py && \
    sed -i 's/from open_spiel.python.games.backgammon.expert_eyes_model import ExpertEyesNet/from expert_eyes_model import ExpertEyesNet/g' /app/trainer_v1.py

# Set the entrypoint
ENTRYPOINT ["python3", "/app/trainer_v1.py"]
