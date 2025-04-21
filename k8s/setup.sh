#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Vietnamese ASR Kubernetes Deployment ===${NC}"

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    echo -e "${RED}Error: doctl is not installed. Please install it first.${NC}"
    echo "See: https://docs.digitalocean.com/reference/doctl/how-to/install/"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed. Please install it first.${NC}"
    echo "See: https://kubernetes.io/docs/tasks/tools/install-kubectl/"
    exit 1
fi

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}Error: terraform is not installed. Please install it first.${NC}"
    echo "See: https://learn.hashicorp.com/tutorials/terraform/install-cli"
    exit 1
fi

# Check if Helm is installed
if ! command -v helm &> /dev/null; then
    echo -e "${RED}Error: helm is not installed. Please install it first.${NC}"
    echo "See: https://helm.sh/docs/intro/install/"
    exit 1
fi
# Get Digital Ocean API token
if [ -z "$DO_API_TOKEN" ]; then
    # Try to load from .env file
    if [ -f "../.env" ]; then
        echo -e "${GREEN}Loading API token from .env file...${NC}"
        DO_API_TOKEN=$(python -c "import os, re; f=open('../.env'); token=re.search(r'DO_API_TOKEN=(.*)', f.read()); print(token.group(1) if token else '')")
    fi

    if [ -f "./.env" ]; then
        echo -e "${GREEN}Loading API token from .env file...${NC}"
        DO_API_TOKEN=$(python -c "import os, re; f=open('./.env'); token=re.search(r'DO_API_TOKEN=(.*)', f.read()); print(token.group(1) if token else '')")
    fi

    # If still empty, ask for user input
    if [ -z "$DO_API_TOKEN" ]; then
        read -p "Enter your Digital Ocean API token: " DO_API_TOKEN
        if [ -z "$DO_API_TOKEN" ]; then
            echo -e "${RED}Error: API token cannot be empty.${NC}"
            exit 1
        fi
    fi
fi

# Authenticate with Digital Ocean
echo -e "\n${YELLOW}=== Authenticating with Digital Ocean ===${NC}"
doctl auth init --access-token string "$DO_API_TOKEN" --context ASR_DEPLOYMENT

# Create terraform.tfvars
echo -e "\n${YELLOW}=== Setting up Terraform ===${NC}"
cd ../terraform

echo "do_token = \"$DO_API_TOKEN\"" > terraform.tfvars
echo -e "${GREEN}Created terraform.tfvars${NC}"

# Initialize Terraform
echo -e "\n${YELLOW}=== Initializing Terraform ===${NC}"
terraform init

# Apply Terraform configuration
echo -e "\n${YELLOW}=== Creating Kubernetes cluster (this may take 5-10 minutes) ===${NC}"
terraform apply -auto-approve

# Configure kubectl
echo -e "\n${YELLOW}=== Configuring kubectl ===${NC}"
CLUSTER_NAME=$(terraform output -raw cluster_id)
doctl kubernetes cluster kubeconfig save "$CLUSTER_NAME"

echo -e "\n${YELLOW}=== Deploying application to Kubernetes ===${NC}"
cd ../k8s

# Create namespaces
kubectl apply -f monitoring/observability-namespace.yaml
kubectl apply -f base/namespace.yaml

# Deploy the application
kubectl apply -f base/

# Wait for deployments to be ready
echo -e "\n${YELLOW}=== Waiting for API deployment to be ready ===${NC}"
kubectl rollout status deployment/asr-api -n asr-system

echo -e "\n${YELLOW}=== Waiting for UI deployment to be ready ===${NC}"
kubectl rollout status deployment/asr-ui -n asr-system

echo -e "\n${GREEN}=== Application deployed successfully! ===${NC}"

# Ask user if they want to set up monitoring
read -p "Do you want to set up monitoring? (y/n): " setup_monitoring
if [[ "$setup_monitoring" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    ./monitoring-setup.sh
fi

# Get service endpoints
echo -e "\n${YELLOW}=== Service Endpoints ===${NC}"
echo -e "${GREEN}API Service:${NC}"
kubectl get svc asr-api -n asr-system

echo -e "\n${GREEN}UI Service:${NC}"
kubectl get svc asr-ui -n asr-system

echo -e "\n${GREEN}=== Deployment completed! ===${NC}"
echo "Access your application using the EXTERNAL-IP addresses from the services above."
