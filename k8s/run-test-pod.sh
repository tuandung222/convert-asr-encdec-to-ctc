#!/bin/bash
# Script to deploy and connect to a test pod for debugging
set -e

# Define color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Source utility functions
source ./setup_utils.sh

echo -e "${YELLOW}=== Deploying ASR Test Pod ===${NC}"

# Check kubectl is configured
if ! ensure_kubeconfig; then
    exit 1
fi

# Check if pod already exists and delete it
if kubectl -n asr-system get pod asr-test-pod &>/dev/null; then
    echo -e "${YELLOW}Test pod already exists. Deleting it...${NC}"
    kubectl -n asr-system delete pod asr-test-pod
    sleep 5
fi

# Apply the test pod YAML
echo -e "${YELLOW}Creating new test pod...${NC}"
kubectl apply -f asr-test-pod.yaml

# Wait for pod to be ready
echo -e "${YELLOW}Waiting for pod to be ready...${NC}"
kubectl -n asr-system wait --for=condition=Ready pod/asr-test-pod --timeout=120s

echo -e "${GREEN}=== Test Pod Ready ===${NC}"
kubectl -n asr-system get pod asr-test-pod

# Display helpful commands
echo -e "\n${YELLOW}=== Available Commands ===${NC}"
echo -e "${GREEN}Connect to pod shell:${NC}"
echo -e "kubectl -n asr-system exec -it asr-test-pod -- /bin/bash"

echo -e "\n${GREEN}Check Python environment:${NC}"
echo -e "kubectl -n asr-system exec -it asr-test-pod -- python -c 'import sys; print(\"Python version:\", sys.version); print(\"Path:\", sys.path)'"

echo -e "\n${GREEN}Test importing api module:${NC}"
echo -e "kubectl -n asr-system exec -it asr-test-pod -- python -c 'import api; print(\"API module found:\", api.__file__)'"

echo -e "\n${GREEN}List files in container:${NC}"
echo -e "kubectl -n asr-system exec -it asr-test-pod -- ls -la /app"

echo -e "\n${GREEN}Delete pod when done:${NC}"
echo -e "kubectl -n asr-system delete pod asr-test-pod"

echo -e "\n${YELLOW}Pod will automatically terminate after 1 hour if not deleted manually.${NC}"

# Ask if user wants to connect to the pod now
read -p "Do you want to connect to the pod shell now? (y/n): " connect_now
if [[ "$connect_now" == "y" ]]; then
    echo -e "${YELLOW}Connecting to pod shell...${NC}"
    kubectl -n asr-system exec -it asr-test-pod -- /bin/bash
fi 