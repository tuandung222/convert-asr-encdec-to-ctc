# #!/bin/bash
# # Script for deploying the ASR application to the Kubernetes cluster
# set -e  # Exit immediately if any command fails

# # Source utility functions
# source ./setup_utils.sh

# echo -e "${YELLOW}=== Deploying Vietnamese ASR Application to Kubernetes ===${NC}"

# # Check kubectl is configured
# if ! kubectl cluster-info &> /dev/null; then
#     echo -e "${RED}Error: kubectl is not configured or cannot connect to the cluster.${NC}"
#     echo -e "Please run ./2_configure_kubernetes.sh first."
#     exit 1
# fi

# # Create required namespaces
# echo -e "\n${YELLOW}=== Creating namespaces ===${NC}"
# kubectl apply -f monitoring/observability-namespace.yaml
# kubectl apply -f base/namespace.yaml

# # Deploy the core application components
# echo -e "\n${YELLOW}=== Deploying ASR API and UI components ===${NC}"
# kubectl apply -f base/

# # Wait for deployments to become ready
# echo -e "\n${YELLOW}=== Waiting for API deployment to be ready ===${NC}"
# kubectl rollout status deployment/asr-api -n asr-system

# echo -e "\n${YELLOW}=== Waiting for UI deployment to be ready ===${NC}"
# kubectl rollout status deployment/asr-ui -n asr-system

# echo -e "\n${GREEN}=== Application deployed successfully! ===${NC}"

# # Display service information
# echo -e "\n${YELLOW}=== Service Endpoints ===${NC}"
# echo -e "${GREEN}API Service:${NC}"
# kubectl get svc asr-api -n asr-system

# echo -e "\n${GREEN}UI Service:${NC}"
# kubectl get svc asr-ui -n asr-system

# # Ask if user wants to set up monitoring
# read -p "Do you want to set up monitoring (Prometheus, Grafana, Jaeger)? (y/n): " setup_monitoring
# if [[ "$setup_monitoring" =~ ^([yY][eE][sS]|[yY])$ ]]; then
#     # echo -e "\n${YELLOW}=== Setting up monitoring... ===${NC}"
#     ./4_setup_monitoring.sh
# else
#     echo -e "\n${YELLOW}=== Skipping monitoring setup ===${NC}"
#     echo "You can set up monitoring later by running ./4_setup_monitoring.sh"
# fi

# echo -e "\n${GREEN}=== Deployment completed! ===${NC}"
# echo "Access your application using the EXTERNAL-IP addresses from the services above." 