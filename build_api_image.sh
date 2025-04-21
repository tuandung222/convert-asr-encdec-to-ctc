#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building API Docker image...${NC}"
docker build -t asr-fastapi-server:test -f api/Dockerfile .

if [ $? -ne 0 ]; then
    echo -e "${RED}Error building Docker image!${NC}"
    exit 1
fi

echo -e "${GREEN}Image built successfully!${NC}"
echo -e "${YELLOW}Running container to test...${NC}"

# Run the container in the background
CONTAINER_ID=$(docker run -d -p 8000:8000 asr-fastapi-server:test)

echo -e "Container started with ID: $CONTAINER_ID"
echo -e "${YELLOW}Waiting for API to start...${NC}"

# Wait for container to initialize
sleep 10

# Check if API is responding
echo -e "${YELLOW}Testing API health endpoint...${NC}"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)

if [ "$RESPONSE" = "200" ]; then
    echo -e "${GREEN}API is healthy! Received HTTP 200 response.${NC}"
    echo -e "${YELLOW}Container logs:${NC}"
    docker logs $CONTAINER_ID
else
    echo -e "${RED}API health check failed with HTTP code $RESPONSE${NC}"
    echo -e "${YELLOW}Container logs:${NC}"
    docker logs $CONTAINER_ID
fi

# Clean up
echo -e "${YELLOW}Stopping and removing test container...${NC}"
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

echo -e "${GREEN}Test completed!${NC}"
echo -e "If the test was successful, you can push the image to your registry with:"
echo -e "docker tag asr-fastapi-server:test tuandung12092002/asr-fastapi-server:latest"
echo -e "docker push tuandung12092002/asr-fastapi-server:latest" 