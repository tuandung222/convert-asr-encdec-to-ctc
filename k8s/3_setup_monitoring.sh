#!/bin/bash
# Script for setting up monitoring with Prometheus, Grafana, and Jaeger
# This now runs BEFORE deploying the application components
set -e  # Exit immediately if any command fails

# Source utility functions
source ./setup_utils.sh

echo -e "${YELLOW}=== Setting up monitoring for Vietnamese ASR on Kubernetes ===${NC}"

# Check kubectl is configured using our utility function
if ! ensure_kubeconfig; then
    exit 1
fi

# Check Digital Ocean load balancer limits
echo -e "\n${YELLOW}=== Checking Digital Ocean resources ===${NC}"
echo -e "Note: Digital Ocean Free Tier has a limit of 2 Load Balancers."
echo -e "All services are configured to use ClusterIP by default."
echo -e "To access services, you should set up the Ingress controller after deployment."
echo -e "Run ./ingress/setup-ingress.sh after completing the deployment."

# Create required namespaces: monitoring and observability
echo -e "\n${YELLOW}=== Creating namespaces ===${NC}"
kubectl apply -f monitoring/namespace.yaml
kubectl create namespace asr-system --dry-run=client -o yaml | kubectl apply -f -

# Deploy Prometheus
echo -e "\n${YELLOW}=== Deploying Prometheus ===${NC}"
kubectl apply -f monitoring/prometheus-configmap.yaml
kubectl apply -f monitoring/prometheus-deployment.yaml
kubectl apply -f monitoring/prometheus-service.yaml

# Deploy Grafana
echo -e "\n${YELLOW}=== Deploying Grafana ===${NC}"
kubectl apply -f monitoring/grafana-configmap.yaml
kubectl apply -f monitoring/grafana-deployment.yaml
kubectl apply -f monitoring/grafana-service.yaml

# Deploy Jaeger
echo -e "\n${YELLOW}=== Deploying Jaeger ===${NC}"
kubectl apply -f monitoring/jaeger-instance.yaml

# Wait for deployments to be ready
echo -e "\n${YELLOW}=== Waiting for Prometheus to be ready ===${NC}"
kubectl -n monitoring rollout status deployment/prometheus --timeout=120s || true

echo -e "\n${YELLOW}=== Waiting for Grafana to be ready ===${NC}"
kubectl -n monitoring rollout status deployment/grafana --timeout=120s || true

echo -e "\n${YELLOW}=== Waiting for Jaeger to be ready ===${NC}"
# Set a longer timeout for Jaeger and handle timeout case
if ! kubectl -n observability rollout status deployment/jaeger --timeout=240s; then
    echo -e "\n${YELLOW}=== Jaeger is taking longer than expected to start ===${NC}"
    echo -e "Checking Jaeger pod status..."
    kubectl -n observability get pods -l app=jaeger
    
    # Get pod name
    JAEGER_POD=$(kubectl -n observability get pods -l app=jaeger -o name | head -1)
    
    if [ -n "$JAEGER_POD" ]; then
        echo -e "\n${YELLOW}=== Jaeger Pod Conditions ===${NC}"
        kubectl -n observability get $JAEGER_POD -o jsonpath='{.status.conditions}' | jq .
        
        echo -e "\n${YELLOW}=== Jaeger Container Status ===${NC}"
        kubectl -n observability get $JAEGER_POD -o jsonpath='{.status.containerStatuses}' | jq .
        
        echo -e "\n${YELLOW}=== Jaeger Logs ===${NC}"
        kubectl -n observability logs $JAEGER_POD --tail=20
    fi
    
    echo -e "\n${YELLOW}=== Continuing deployment despite Jaeger delay ===${NC}"
    echo -e "You can check Jaeger status later with: kubectl -n observability get pods"
fi

# Check if Grafana is having issues and provide common fixes
if kubectl -n monitoring get pods | grep grafana | grep -E "0/1|Error|CrashLoopBackOff" > /dev/null; then
    echo -e "\n${YELLOW}=== Grafana Troubleshooting ===${NC}"
    echo -e "Grafana pod is not starting properly. Here are some common fixes:"
    echo -e "1. Check if it's a resource issue:"
    echo -e "   kubectl -n monitoring describe pod \$(kubectl -n monitoring get pods -l app=grafana -o name | head -1) | grep -A5 Events"
    
    echo -e "\n2. If it's a permission issue, you can try running Grafana as root:"
    echo -e "   kubectl -n monitoring patch deployment grafana -p '{\"spec\":{\"template\":{\"spec\":{\"securityContext\":{\"runAsUser\":0,\"runAsGroup\":0,\"fsGroup\":0}}}}}'"
    
    echo -e "\n3. Inspect Grafana logs to find the specific issue:"
    echo -e "   kubectl -n monitoring logs \$(kubectl -n monitoring get pods -l app=grafana -o name)"
    
    echo -e "\nYou can proceed with the deployment and fix Grafana later. The API will still work without Grafana."
fi

# Display information about the deployed monitoring services
echo -e "\n${YELLOW}=== Monitoring Services Deployed ===${NC}"
echo -e "Services have been deployed with ClusterIP type and are not directly accessible."
echo -e "After completing the deployment, set up the Ingress controller to access all services:"
echo -e "${GREEN}./ingress/setup-ingress.sh${NC}"
echo -e "This will create a single ingress point for all services using a single LoadBalancer."

echo -e "\n${GREEN}=== Monitoring setup completed! ===${NC}"
echo -e "Next step: Run ./4_deploy_application.sh to deploy the ASR application" 