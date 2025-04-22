#!/bin/bash
# Script for setting up the infrastructure on Digital Ocean with Terraform
set -e  # Exit immediately if any command fails

# Source utility functions
source ./setup_utils.sh

echo -e "${YELLOW}=== Vietnamese ASR Kubernetes Infrastructure Setup ===${NC}"

# Check prerequisites
check_prerequisites || exit 1

# Show security warning
display_security_warning

# Get DigitalOcean API token
get_api_token || exit 1

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
    echo "kubernetes_version = \"1.32.2-do.0\"" >> terraform.tfvars
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

# Get cluster information
CLUSTER_ID=$(terraform output -raw cluster_id)

echo -e "\n${GREEN}=== Infrastructure setup completed! ===${NC}"
echo "Cluster ID: $CLUSTER_ID"
echo -e "\nNext step: Run ./2_configure_kubernetes.sh to configure kubectl" 