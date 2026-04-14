#!/bin/bash

# Configuration
# Stop the script if any command fails
set -e

PROJECT_ID="expert-eyes-training-742"
REPO_NAME="expert-eyes-repo"
LOCATION="us-central1"
IMAGE_NAME="trainer"
IMAGE_TAG="v7"
FULL_IMAGE_NAME="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Ensure gcloud is in the PATH (found in /home/jeremy/google-cloud-sdk/bin)
export PATH="/home/jeremy/google-cloud-sdk/bin:$PATH"

echo "=== Starting Cloud-Native Build & Push via Google Cloud Build (Region: ${LOCATION}) ==="
echo "Target Image: ${FULL_IMAGE_NAME}"

# Verify gcloud availability
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud command not found even after updating PATH."
    exit 1
fi

# We use '--region' to ensure the build request is routed to the correct regional service.
# We also use 'gcloud builds submit' to handle remote building and 
# automatic pushing to Artifact Registry.
gcloud builds submit --region=${LOCATION} --tag ${FULL_IMAGE_NAME} .

echo "=== Success! Image is now in Artifact Registry ==="
