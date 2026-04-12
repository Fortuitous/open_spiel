# Use Python 3.12 slim for compatibility (OpenSpiel requires >=3.11)
FROM python:3.12-slim

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

# Copy the entire source tree to the container
COPY . .

# Install Python dependencies
# Note: We install torch first to ensure it's prioritized
RUN pip install --no-cache-dir torch google-cloud-storage

# Install OpenSpiel from local source to ensure C++ modifications are included
RUN pip install .

# Patch trainer_v1.py to remove local absolute paths and fix imports
# The script is located at open_spiel/python/games/backgammon/trainer_v1.py
# We will move it to the root of /app for easier execution
RUN cp open_spiel/python/games/backgammon/trainer_v1.py /app/trainer_v1.py && \
    cp open_spiel/python/games/backgammon/expert_eyes_model.py /app/expert_eyes_model.py && \
    sed -i '/sys.path.append("\/home\/jeremy\/projects\/open_spiel")/d' /app/trainer_v1.py && \
    sed -i '/sys.path.append("\/home\/jeremy\/projects\/open_spiel\/build\/python")/d' /app/trainer_v1.py && \
    sed -i 's/from open_spiel.python.games.backgammon.expert_eyes_model import ExpertEyesNet/from expert_eyes_model import ExpertEyesNet/g' /app/trainer_v1.py

# Set the entrypoint to run the trainer
ENTRYPOINT ["python3", "/app/trainer_v1.py"]
