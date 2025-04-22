#!/bin/bash
# Script for cleaning up Kubernetes resources and Digital Ocean infrastructure
set -e  # Exit immediately if any command fails

# Source utility functions
source ./setup_utils.sh

echo -e "${YELLOW}=== Cleaning up Vietnamese ASR Kubernetes Deployment ===${NC}"
echo -e "${RED}WARNING: This will destroy all resources and data!${NC}"
read -p "Are you sure you want to continue? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Get DigitalOcean API token if not already set
get_api_token || exit 1

# DigitalOcean authentication if needed
if ! doctl account get &> /dev/null; then
    echo -e "\n${YELLOW}=== Authenticating with Digital Ocean ===${NC}"
    doctl auth init --access-token string "$DO_API_TOKEN" --context ASR_DEPLOYMENT
    doctl auth switch --context ASR_DEPLOYMENT
fi

# Delete application resources first
echo -e "\n${YELLOW}=== Deleting ASR application resources ===${NC}"
kubectl delete namespace asr-system --ignore-not-found=true

# Delete monitoring resources
echo -e "\n${YELLOW}=== Deleting monitoring resources ===${NC}"
kubectl delete namespace monitoring --ignore-not-found=true
kubectl delete namespace observability --ignore-not-found=true

# Destroy Terraform-managed infrastructure
echo -e "\n${YELLOW}=== Destroying Digital Ocean Kubernetes cluster ===${NC}"
cd ../terraform

# Check if terraform state exists
if [ ! -f "terraform.tfstate" ]; then
    echo -e "${YELLOW}Warning: Terraform state file not found. Infrastructure may have already been destroyed.${NC}"
    read -p "Do you want to continue with the cleanup process? (y/n): " continue
    if [[ "$continue" != "y" ]]; then
        echo "Operation cancelled."
        exit 0
    fi
    # If continuing without state file, we'll skip the terraform destroy step
else
    # Get cluster ID for verification
    CLUSTER_ID=$(terraform output -raw cluster_id 2>/dev/null || echo "unknown")
    echo -e "${YELLOW}Preparing to destroy cluster: $CLUSTER_ID${NC}"
    
    # Destroy the infrastructure
    terraform destroy -auto-approve
    echo -e "${GREEN}Infrastructure destroyed successfully.${NC}"
fi

# Clean up local files
echo -e "\n${YELLOW}=== Cleaning up local files ===${NC}"
rm -f terraform.tfvars

echo -e "\n${GREEN}=== Cleanup completed! ===${NC}"
echo "All resources have been deleted." 