#!/bin/bash
# This script automates the deployment of a Vietnamese ASR (Automatic Speech Recognition) system on a Kubernetes cluster in DigitalOcean.
# It handles the entire process from authentication to deployment and configuration.
set -e  # Exit immediately if any command fails

# Define color codes for better readability in terminal output
GREEN='\033[0;32m'   # Success messages
YELLOW='\033[1;33m'  # Section headers and warnings
RED='\033[0;31m'     # Error messages
NC='\033[0m'         # No Color (resets formatting)

echo -e "${YELLOW}=== Vietnamese ASR Kubernetes Deployment ===${NC}"

# Prerequisite checks: Verify all required tools are installed before proceeding
# Check if doctl (DigitalOcean CLI) is installed
if ! command -v doctl &> /dev/null; then
    echo -e "${RED}Error: doctl is not installed. Please install it first.${NC}"
    echo "See: https://docs.digitalocean.com/reference/doctl/how-to/install/"
    exit 1
fi

# Check if kubectl (Kubernetes command-line tool) is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed. Please install it first.${NC}"
    echo "See: https://kubernetes.io/docs/tasks/tools/install-kubectl/"
    exit 1
fi

# Check if Terraform (Infrastructure as Code tool) is installed
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}Error: terraform is not installed. Please install it first.${NC}"
    echo "See: https://learn.hashicorp.com/tutorials/terraform/install-cli"
    exit 1
fi

# Check if Helm (Kubernetes package manager) is installed
if ! command -v helm &> /dev/null; then
    echo -e "${RED}Error: helm is not installed. Please install it first.${NC}"
    echo "See: https://helm.sh/docs/intro/install/"
    exit 1
fi

# API token acquisition: Try multiple sources to get the DigitalOcean API token
if [ -z "$DO_API_TOKEN" ]; then
    # First, try to load from parent directory .env file
    if [ -f "../.env" ]; then
        echo -e "${GREEN}Loading API token from .env file...${NC}"
        export $(grep -v '^#' ../.env | xargs)  # Export all non-commented variables
    fi

    # Then, try to load from current directory .env file
    if [ -f "./.env" ]; then
        echo -e "${GREEN}Loading API token from .env file...${NC}"
        export $(grep -v '^#' ./.env | xargs)
    fi

    # If token is still not set, prompt the user for manual input
    if [ -z "$DO_API_TOKEN" ]; then
        read -p "Enter your Digital Ocean API token: " DO_API_TOKEN
        if [ -z "$DO_API_TOKEN" ]; then
            echo -e "${RED}Error: API token cannot be empty.${NC}"
            exit 1
        fi
    fi
fi

# DigitalOcean authentication: Set up CLI authentication with the API token
echo -e "\n${YELLOW}=== Authenticating with Digital Ocean ===${NC}"
doctl auth init --access-token string "$DO_API_TOKEN" --context ASR_DEPLOYMENT  # Initialize authentication
doctl auth switch --context ASR_DEPLOYMENT  # Switch to the created context

# Terraform configuration: Set up variables for infrastructure provisioning
echo -e "\n${YELLOW}=== Setting up Terraform ===${NC}"
cd ../terraform

# Create terraform.tfvars file with configuration values
if [ -f "terraform.tfvars.example" ]; then
    # If an example file exists, use it as a template
    cp terraform.tfvars.example terraform.tfvars
    sed -i "s|your-digitalocean-api-token|$DO_API_TOKEN|g" terraform.tfvars
else
    # Otherwise, create a new file with default values
    echo "do_token = \"$DO_API_TOKEN\"" > terraform.tfvars
    echo "region = \"sgp1\"" >> terraform.tfvars  # Singapore region
    echo "kubernetes_version = \"1.29.1-do.0\"" >> terraform.tfvars  # Kubernetes version
    echo "node_size = \"s-2vcpu-4gb\"" >> terraform.tfvars  # Node size (2 vCPU, 4GB RAM)
    echo "node_count = 3" >> terraform.tfvars  # Number of worker nodes
fi
echo -e "${GREEN}Created terraform.tfvars${NC}"

# Initialize Terraform workspace
echo -e "\n${YELLOW}=== Initializing Terraform ===${NC}"
terraform init  # Download providers and set up backend

# Create infrastructure: Apply Terraform configuration to provision the Kubernetes cluster
echo -e "\n${YELLOW}=== Creating Kubernetes cluster (this may take 5-10 minutes) ===${NC}"
terraform apply -auto-approve  # Create resources without confirmation prompt

# Configure kubectl to communicate with the new cluster
echo -e "\n${YELLOW}=== Configuring kubectl ===${NC}"
CLUSTER_NAME=$(terraform output -raw cluster_id)  # Get cluster ID from Terraform output
doctl kubernetes cluster kubeconfig save "$CLUSTER_NAME"  # Save kubeconfig file

# Application deployment: Deploy the ASR application to the Kubernetes cluster
echo -e "\n${YELLOW}=== Deploying application to Kubernetes ===${NC}"
cd ../k8s

# Create required Kubernetes namespaces
kubectl apply -f monitoring/observability-namespace.yaml  # Namespace for monitoring tools
kubectl apply -f base/namespace.yaml  # Namespace for the ASR application

# Deploy the core application components
kubectl apply -f base/  # Apply all manifests in the base directory

# Wait for deployments to become ready
echo -e "\n${YELLOW}=== Waiting for API deployment to be ready ===${NC}"
kubectl rollout status deployment/asr-api -n asr-system  # Wait for API deployment

echo -e "\n${YELLOW}=== Waiting for UI deployment to be ready ===${NC}"
kubectl rollout status deployment/asr-ui -n asr-system  # Wait for UI deployment

echo -e "\n${GREEN}=== Application deployed successfully! ===${NC}"

# Optional monitoring setup: Ask user if they want to install monitoring tools
read -p "Do you want to set up monitoring? (y/n): " setup_monitoring
if [[ "$setup_monitoring" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    ./monitoring-setup.sh  # Run the monitoring setup script
fi

# Display service information: Show how to access the deployed services
echo -e "\n${YELLOW}=== Service Endpoints ===${NC}"
echo -e "${GREEN}API Service:${NC}"
kubectl get svc asr-api -n asr-system  # Display API service details (including external IP)

echo -e "\n${GREEN}UI Service:${NC}"
kubectl get svc asr-ui -n asr-system  # Display UI service details (including external IP)

echo -e "\n${GREEN}=== Deployment completed! ===${NC}"
echo "Access your application using the EXTERNAL-IP addresses from the services above."
