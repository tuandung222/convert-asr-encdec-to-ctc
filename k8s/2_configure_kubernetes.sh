#!/bin/bash
# Script for configuring kubectl with the newly created cluster
set -e  # Exit immediately if any command fails

# Source utility functions
source ./setup_utils.sh

echo -e "${YELLOW}=== Configuring Kubernetes for Vietnamese ASR Deployment ===${NC}"

# Check prerequisites
if ! command -v kubectl &> /dev/null || ! command -v doctl &> /dev/null; then
    echo -e "${RED}Error: kubectl and doctl are required for this script.${NC}"
    exit 1
fi

# Get DigitalOcean API token if not already set
get_api_token || exit 1

# DigitalOcean authentication if needed
if ! doctl account get &> /dev/null; then
    echo -e "\n${YELLOW}=== Authenticating with Digital Ocean ===${NC}"
    doctl auth init --access-token string "$DO_API_TOKEN" --context ASR_DEPLOYMENT
    doctl auth switch --context ASR_DEPLOYMENT
fi

# Get cluster ID from Terraform output
cd ../terraform
if [ ! -f "terraform.tfstate" ]; then
    echo -e "${RED}Error: Terraform state file not found. Please run 1_infrastructure_setup.sh first.${NC}"
    exit 1
fi

CLUSTER_ID=$(terraform output -raw cluster_id)
if [ -z "$CLUSTER_ID" ]; then
    echo -e "${RED}Error: Could not get cluster ID from Terraform output.${NC}"
    exit 1
fi

# Configure kubectl to communicate with the cluster
echo -e "\n${YELLOW}=== Configuring kubectl for cluster $CLUSTER_ID ===${NC}"

# Save kubeconfig to a secure location
KUBECONFIG_DIR="${HOME}/.kube"
mkdir -p "$KUBECONFIG_DIR"
export KUBECONFIG="${KUBECONFIG_DIR}/config-${CLUSTER_ID}"
echo -e "Saving kubeconfig to: ${KUBECONFIG}"
# doctl kubernetes cluster kubeconfig save "$CLUSTER_ID" --set-current-context
doctl kubernetes cluster kubeconfig save "$CLUSTER_ID" 

# Verify connection
echo -e "\n${YELLOW}=== Verifying connection to Kubernetes cluster ===${NC}"
if kubectl cluster-info; then
    echo -e "\n${GREEN}=== Kubernetes configuration successful! ===${NC}"
    echo -e "You can now use kubectl to manage your cluster."
    echo -e "\nNext step: Run ./3_setup_monitoring.sh to set up monitoring"
else
    echo -e "\n${RED}=== Failed to connect to Kubernetes cluster. ===${NC}"
    echo -e "Please check your Digital Ocean account and try again."
    exit 1
fi 