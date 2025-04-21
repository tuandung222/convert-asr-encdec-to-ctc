#!/bin/bash

# Get tag from argument or use default value
TAG=${1:-latest}

# Build main image
docker build -t speech-api:$TAG -f Dockerfile .

# Build UI image
docker build -t speech-ui:$TAG -f ui/Dockerfile ui/

# Build Gradio image
# docker build -t speech-gradio:$TAG -f src/app/Dockerfile.gradio .

echo "Successfully built images with tag $TAG"
