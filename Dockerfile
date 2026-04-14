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

# Install OpenSpiel dependencies first
COPY setup.py MANIFEST.in requirements.txt README.md /app/
COPY open_spiel /app/open_spiel
COPY pybind11 /app/pybind11

# Build and install OpenSpiel
# This is the heavy layer that will be cached if the above files don't change
RUN pip install .

# Copy the specific trainer scripts LAST
# These are the files we edit frequently; copying them here ensures near-instant rebuilds
COPY open_spiel/python/games/backgammon/trainer_v1.py /app/trainer_v1.py
COPY open_spiel/python/games/backgammon/expert_eyes_model.py /app/expert_eyes_model.py

# Patch the local paths in the scripts
RUN sed -i '/sys.path.append("\/home\/jeremy\/projects\/open_spiel")/d' /app/trainer_v1.py && \
    sed -i '/sys.path.append("\/home\/jeremy\/projects\/open_spiel\/build\/python")/d' /app/trainer_v1.py && \
    sed -i 's/from open_spiel.python.games.backgammon.expert_eyes_model import ExpertEyesNet/from expert_eyes_model import ExpertEyesNet/g' /app/trainer_v1.py

# Set the entrypoint
ENTRYPOINT ["python3", "/app/trainer_v1.py"]
