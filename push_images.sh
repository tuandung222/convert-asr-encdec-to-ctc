#!/bin/bash

# Configuration
REGISTRY="tuandung12092002"
API_IMAGE="asr-fastapi-server"
UI_IMAGE="asr-streamlit-ui"
TAG=$(date +"%Y%m%d_%H%M%S")
LATEST_TAG="latest"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Building and pushing ASR Docker images to ${REGISTRY} ===${NC}"
echo -e "API Image: ${REGISTRY}/${API_IMAGE}"
echo -e "UI Image: ${REGISTRY}/${UI_IMAGE}"
echo -e "Version Tag: ${TAG}"

# Build API image
echo -e "\n${YELLOW}=== Building API image... ===${NC}"
docker build -t ${REGISTRY}/${API_IMAGE}:${TAG} \
  -t ${REGISTRY}/${API_IMAGE}:${LATEST_TAG} \
  --build-arg APP_USER=api \
  --build-arg APP_USER_UID=1000 \
  -f api/Dockerfile .

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ API image build failed!${NC}"
  exit 1
fi
echo -e "${GREEN}✅ API image built successfully${NC}"

# Build UI image
echo -e "\n${YELLOW}=== Building UI image... ===${NC}"
docker build -t ${REGISTRY}/${UI_IMAGE}:${TAG} \
  -t ${REGISTRY}/${UI_IMAGE}:${LATEST_TAG} \
  --build-arg APP_USER=streamlit \
  --build-arg APP_USER_UID=1000 \
  -f ui/Dockerfile .

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ UI image build failed!${NC}"
  exit 1
fi
echo -e "${GREEN}✅ UI image built successfully${NC}"

# Login to Docker Hub
echo -e "\n${YELLOW}=== Logging in to Docker registry ${REGISTRY}... ===${NC}"
docker login

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Docker login failed!${NC}"
  exit 1
fi
echo -e "${GREEN}✅ Successfully logged in to Docker registry${NC}"

# Push API image
echo -e "\n${YELLOW}=== Pushing API image to registry... ===${NC}"
docker push ${REGISTRY}/${API_IMAGE}:${TAG}
docker push ${REGISTRY}/${API_IMAGE}:${LATEST_TAG}

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ API image push failed!${NC}"
  exit 1
fi
echo -e "${GREEN}✅ API image pushed successfully${NC}"

# Push UI image
echo -e "\n${YELLOW}=== Pushing UI image to registry... ===${NC}"
docker push ${REGISTRY}/${UI_IMAGE}:${TAG}
docker push ${REGISTRY}/${UI_IMAGE}:${LATEST_TAG}

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ UI image push failed!${NC}"
  exit 1
fi
echo -e "${GREEN}✅ UI image pushed successfully${NC}"

# Summary
echo -e "\n${GREEN}=== All images built and pushed successfully! ===${NC}"
echo -e "API Image: ${REGISTRY}/${API_IMAGE}:${TAG}"
echo -e "UI Image: ${REGISTRY}/${UI_IMAGE}:${TAG}"
echo -e "Also tagged as: ${LATEST_TAG}"
echo -e "\nTo run the images:"
echo -e "docker run -p 8000:8000 ${REGISTRY}/${API_IMAGE}:${LATEST_TAG}"
echo -e "docker run -p 8501:8501 ${REGISTRY}/${UI_IMAGE}:${LATEST_TAG}"