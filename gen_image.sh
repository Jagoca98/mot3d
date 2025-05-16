#!/bin/bash

# This script is used to build

# Get the current directory
set -e
current_dir=$(pwd)

#  Load the environment variables from the config.env file
set -o allexport
source config.env
set -o allexport

# Ensure environment variables are set
: "${USER_ID:?Need to set USER_ID}"
: "${USER_NAME:?Need to set USER_NAME}"
: "${GROUP_ID:?Need to set GROUP_ID}"
: "${GROUP_NAME:?Need to set GROUP_NAME}"
: "${WORKSPACE:?Need to set WORKSPACE}"
: "${DOCKER_IMAGE_NAME:?Need to set DOCKER_IMAGE_NAME}"


# Remove the existing Docker container if it exists
if docker ps -a | grep $DOCKER_IMAGE_NAME; then
    docker rm $DOCKER_IMAGE_NAME
fi

# Build Docker image if it does not exist
echo "Building Docker image..."
if docker buildx build \
    --build-arg GROUP_ID=$GROUP_ID \
    --build-arg GROUP_NAME=$GROUP_NAME \
    --build-arg USER_ID=$USER_ID \
    --build-arg USER_NAME=$USER_NAME \
    --build-arg WORKSPACE=$WORKSPACE \
    -t $DOCKER_IMAGE_NAME \
    -f Dockerfile . ; then
    echo "Docker image built successfully."
else
    echo "Error: Failed to build Docker image."
    exit 1
fi
