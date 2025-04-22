#!/bin/bash
# This script handles DigitalOcean authentication and Terraform infrastructure setup
set -e  # Exit immediately if any command fails

# Define color codes for better readability in terminal output
GREEN='\033[0;32m'   # Success messages
YELLOW='\033[1;33m'  # Section headers and warnings
RED='\033[0;31m'     # Error messages
NC='\033[0m'         # No Color (resets formatting)

echo -e "${YELLOW}=== Vietnamese ASR Kubernetes Infrastructure Setup ===${NC}"

# Prerequisite checks for required tools
if ! command -v doctl &> /dev/null; then
    echo -e "${RED}Error: doctl is not installed. Please install it first.${NC}"
    echo "See: https://docs.digitalocean.com/reference/doctl/how-to/install/"
    exit 1
fi

if ! command -v terraform &> /dev/null; then
    echo -e "${RED}Error: terraform is not installed. Please install it first.${NC}"
    echo "See: https://learn.hashicorp.com/tutorials/terraform/install-cli"
    exit 1
fi

# API token acquisition
if [ -z "$DO_API_TOKEN" ]; then
    # Try to load from parent directory .env file
    if [ -f "../.env" ]; then
        echo -e "${GREEN}Loading API token from .env file...${NC}"
        export $(grep -v '^#' ../.env | xargs)
    fi

    # Try to load from current directory .env file
    if [ -f "./.env" ]; then
        echo -e "${GREEN}Loading API token from .env file...${NC}"
        export $(grep -v '^#' ./.env | xargs)
    fi

    # If token is still not set, prompt the user
    if [ -z "$DO_API_TOKEN" ]; then
        read -p "Enter your Digital Ocean API token: " DO_API_TOKEN
        if [ -z "$DO_API_TOKEN" ]; then
            echo -e "${RED}Error: API token cannot be empty.${NC}"
            exit 1
        fi
    fi
fi

# DigitalOcean authentication
echo -e "\n${YELLOW}=== Authenticating with Digital Ocean ===${NC}"
doctl auth init --access-token string "$DO_API_TOKEN" --context ASR_DEPLOYMENT
doctl auth switch --context ASR_DEPLOYMENT

# Terraform configuration
echo -e "\n${YELLOW}=== Setting up Terraform ===${NC}"
cd ../terraform

# Create terraform.tfvars file
if [ -f "terraform.tfvars.example" ]; then
    cp terraform.tfvars.example terraform.tfvars
    sed -i "s|your-digitalocean-api-token|$DO_API_TOKEN|g" terraform.tfvars
else
    echo "do_token = \"$DO_API_TOKEN\"" > terraform.tfvars
    echo "region = \"sgp1\"" >> terraform.tfvars
    echo "kubernetes_version = \"1.29.1-do.0\"" >> terraform.tfvars
    echo "node_size = \"s-2vcpu-4gb\"" >> terraform.tfvars
    echo "node_count = 3" >> terraform.tfvars
fi
echo -e "${GREEN}Created terraform.tfvars${NC}"

# Initialize Terraform workspace
echo -e "\n${YELLOW}=== Initializing Terraform ===${NC}"
terraform init

# Create infrastructure
echo -e "\n${YELLOW}=== Creating Kubernetes cluster (this may take 5-10 minutes) ===${NC}"
terraform apply -auto-approve

echo -e "\n${GREEN}=== Infrastructure setup completed! ===${NC}"
echo "Cluster ID: $(terraform output -raw cluster_id)"
echo -e "\nRun deploy_api_ui.sh next to deploy the application components."

# Make the deploy script executable
chmod +x ../k8s/deploy_api_ui.sh
