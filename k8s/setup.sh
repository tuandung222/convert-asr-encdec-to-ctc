#!/bin/bash
# Main setup script for Vietnamese ASR Kubernetes deployment
# This orchestrates the individual scripts for a complete deployment
set -e  # Exit immediately if any command fails

# Define color codes for better readability in terminal output
GREEN='\033[0;32m'   # Success messages
YELLOW='\033[1;33m'  # Section headers and warnings
RED='\033[0;31m'     # Error messages
NC='\033[0m'         # No Color (resets formatting)

# Source utility functions
source ./setup_utils.sh

echo -e "${YELLOW}=====================================================${NC}"
echo -e "${YELLOW}=== Vietnamese ASR Kubernetes Deployment - Main Setup ===${NC}"
echo -e "${YELLOW}=====================================================${NC}"

# Make all scripts executable
make_scripts_executable "setup_utils.sh" "1_infrastructure_setup.sh" "2_configure_kubernetes.sh" "3_setup_monitoring.sh" "4_deploy_application.sh" "5_cleanup.sh"

# Check if the user wants a guided deployment
echo -e "This script will guide you through the deployment process."
echo -e "You can run each step individually or let this script run them all in sequence."
read -p "Do you want to run the complete deployment process? (y/n): " run_all

if [[ "$run_all" != "y" ]]; then
    echo -e "\n${YELLOW}Available steps:${NC}"
    echo "1. ./1_infrastructure_setup.sh - Create Kubernetes cluster on Digital Ocean"
    echo "2. ./2_configure_kubernetes.sh - Configure kubectl for the cluster"
    echo "3. ./3_setup_monitoring.sh - Set up monitoring stack (Prometheus, Grafana, Jaeger)"
    echo "4. ./4_deploy_application.sh - Deploy the ASR application (API and UI)"
    echo "5. ./5_cleanup.sh - Clean up all resources when done"
    echo -e "\nRun each script individually when you're ready."
    exit 0
fi

# Run each step in sequence
echo -e "\n${YELLOW}=== Starting complete deployment process ===${NC}"

# Step 1: Infrastructure setup
echo -e "\n${YELLOW}=== Step 1: Setting up infrastructure ===${NC}"
./1_infrastructure_setup.sh
echo -e "\n${GREEN}=== Infrastructure setup completed! ===${NC}"

# Step 2: Configure Kubernetes
echo -e "\n${YELLOW}=== Step 2: Configuring Kubernetes ===${NC}"
source ./2_configure_kubernetes.sh
echo -e "\n${GREEN}=== Kubernetes configuration completed! ===${NC}"

# Make sure KUBECONFIG is preserved for next steps
if [ -f ../terraform/terraform.tfstate ]; then
    cd ../terraform
    CLUSTER_ID=$(terraform output -raw cluster_id 2>/dev/null || echo "")
    cd ../k8s
    if [ ! -z "$CLUSTER_ID" ]; then
        export KUBECONFIG="${HOME}/.kube/config-${CLUSTER_ID}"
        echo -e "${GREEN}KUBECONFIG set to: ${KUBECONFIG}${NC}"
        
        # Verify kubernetes connection
        if ! kubectl cluster-info &>/dev/null; then
            echo -e "${RED}Error: kubectl is not configured properly. Trying to reconnect...${NC}"
            doctl kubernetes cluster kubeconfig save "$CLUSTER_ID" --set-current-context
            
            if ! kubectl cluster-info &>/dev/null; then
                echo -e "${RED}Failed to configure kubectl. Please run setup step by step manually.${NC}"
                exit 1
            fi
        fi
    fi
fi

# Step 3: Setup monitoring (now before deploying application)
echo -e "\n${YELLOW}=== Step 3: Setting up monitoring ===${NC}"
source ./3_setup_monitoring.sh
echo -e "\n${GREEN}=== Monitoring setup completed! ===${NC}"

# Step 4: Deploy application
echo -e "\n${YELLOW}=== Step 4: Deploying application ===${NC}"
source ./4_deploy_application.sh
echo -e "\n${GREEN}=== Application deployment completed! ===${NC}"

echo -e "\n${GREEN}=====================================================${NC}"
echo -e "${GREEN}=== Vietnamese ASR Kubernetes Deployment Completed! ===${NC}"
echo -e "${GREEN}=====================================================${NC}"

echo -e "\nWhen you're done with the deployment, you can clean up all resources with:"
echo -e "./5_cleanup.sh"
