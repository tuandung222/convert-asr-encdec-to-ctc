#!/bin/bash
# This script deploys the Vietnamese ASR API and UI components to a Kubernetes cluster
set -e  # Exit immediately if any command fails

# Define color codes for better readability in terminal output
GREEN='\033[0;32m'   # Success messages
YELLOW='\033[1;33m'  # Section headers and warnings
RED='\033[0;31m'     # Error messages
NC='\033[0m'         # No Color (resets formatting)

echo -e "${YELLOW}=== Vietnamese ASR Application Deployment ===${NC}"

# Prerequisite checks
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed. Please install it first.${NC}"
    echo "See: https://kubernetes.io/docs/tasks/tools/install-kubectl/"
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo -e "${RED}Error: helm is not installed. Please install it first.${NC}"
    echo "See: https://helm.sh/docs/intro/install/"
    exit 1
fi

# Check for terraform output to get cluster ID
if [ ! -f "../terraform/terraform.tfstate" ]; then
    echo -e "${RED}Error: Terraform state file not found. Please run terraform-digital-ocean-setup.sh first.${NC}"
    exit 1
fi

# Configure kubectl to communicate with the cluster
echo -e "\n${YELLOW}=== Configuring kubectl ===${NC}"
cd ../terraform
CLUSTER_NAME=$(terraform output -raw cluster_id)
doctl kubernetes cluster kubeconfig save "$CLUSTER_NAME"

# Deploy application components
echo -e "\n${YELLOW}=== Deploying application to Kubernetes ===${NC}"
cd ../k8s

# Create required namespaces
kubectl apply -f monitoring/observability-namespace.yaml
kubectl apply -f base/namespace.yaml

# Deploy the core application components
kubectl apply -f base/

# Wait for deployments to become ready
echo -e "\n${YELLOW}=== Waiting for API deployment to be ready ===${NC}"
kubectl rollout status deployment/asr-api -n asr-system

echo -e "\n${YELLOW}=== Waiting for UI deployment to be ready ===${NC}"
kubectl rollout status deployment/asr-ui -n asr-system

echo -e "\n${GREEN}=== Application deployed successfully! ===${NC}"

# Optional monitoring setup
read -p "Do you want to set up monitoring? (y/n): " setup_monitoring
if [[ "$setup_monitoring" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    ./monitoring-setup.sh
fi

# Display service information
echo -e "\n${YELLOW}=== Service Endpoints ===${NC}"
echo -e "${GREEN}API Service:${NC}"
kubectl get svc asr-api -n asr-system

echo -e "\n${GREEN}UI Service:${NC}"
kubectl get svc asr-ui -n asr-system

echo -e "\n${GREEN}=== Deployment completed! ===${NC}"
echo "Access your application using the EXTERNAL-IP addresses from the services above."
