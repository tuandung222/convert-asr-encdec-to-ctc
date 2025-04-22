#!/bin/bash
# Script for deploying the ASR application to the Kubernetes cluster
# This now runs AFTER setting up monitoring
set -e  # Exit immediately if any command fails

# Source utility functions
source ./setup_utils.sh

echo -e "${YELLOW}=== Deploying Vietnamese ASR Application to Kubernetes ===${NC}"

# Check kubectl is configured using our utility function
if ! ensure_kubeconfig; then
    exit 1
fi

# Check if monitoring namespace exists
if ! kubectl get namespace monitoring &> /dev/null; then
    echo -e "${YELLOW}Warning: Monitoring namespace not found. It's recommended to run ./3_setup_monitoring.sh first.${NC}"
    read -p "Do you want to continue anyway? (y/n): " continue_anyway
    if [[ "$continue_anyway" != "y" ]]; then
        echo "Deployment cancelled. Please run ./3_setup_monitoring.sh first."
        exit 0
    fi
else
    # Check monitoring components status but don't block deployment
    echo -e "\n${YELLOW}=== Checking monitoring status ===${NC}"
    PROMETHEUS_PODS=$(kubectl -n monitoring get pods | grep -v prometheus-grafana | grep prometheus)
    GRAFANA_PODS=$(kubectl -n monitoring get pods | grep grafana)
    
    echo -e "Prometheus components:"
    echo "$PROMETHEUS_PODS"
    
    echo -e "\nGrafana status:"
    echo "$GRAFANA_PODS"
    
    if echo "$GRAFANA_PODS" | grep -E "0/|Error|CrashLoopBackOff" > /dev/null; then
        echo -e "${YELLOW}Warning: Grafana appears to be having issues.${NC}"
        echo -e "This won't prevent the ASR application from working."
        echo -e "You can fix Grafana later by running ./fix-grafana.sh\n"
    fi
fi

# Verify namespaces exist or create if needed
echo -e "\n${YELLOW}=== Verifying namespaces ===${NC}"
kubectl create namespace asr-system --dry-run=client -o yaml | kubectl apply -f -

# Deploy the core application components
echo -e "\n${YELLOW}=== Deploying ASR API and UI components ===${NC}"
kubectl apply -f base/

# Wait for deployments to become ready
echo -e "\n${YELLOW}=== Waiting for API deployment to be ready ===${NC}"
kubectl rollout status deployment/asr-api -n asr-system

echo -e "\n${YELLOW}=== Waiting for UI deployment to be ready ===${NC}"
kubectl rollout status deployment/asr-ui -n asr-system

echo -e "\n${GREEN}=== Application deployed successfully! ===${NC}"

# Display service information
echo -e "\n${YELLOW}=== Service Endpoints ===${NC}"
echo -e "${GREEN}API Service:${NC}"
kubectl get svc asr-api -n asr-system

echo -e "\n${GREEN}UI Service:${NC}"
kubectl get svc asr-ui -n asr-system

# Display all available services including monitoring
echo -e "\n${YELLOW}=== All Available Services ===${NC}"

# API and UI
echo -e "\n${GREEN}ASR Services:${NC}"
kubectl get svc -n asr-system

# Monitoring services
if kubectl get namespace monitoring &> /dev/null; then
    echo -e "\n${GREEN}Monitoring Services:${NC}"
    kubectl get svc -n monitoring
    
    echo -e "\n${GREEN}Grafana:${NC}"
    echo "Username: admin"
    echo "Password: admin"
fi

# Jaeger services
if kubectl get namespace observability &> /dev/null; then
    echo -e "\n${GREEN}Jaeger Services:${NC}"
    kubectl get svc -n observability
fi

echo -e "\n${GREEN}=== Deployment completed! ===${NC}"
echo "Access your application using the EXTERNAL-IP addresses from the services above." 