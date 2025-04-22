#!/bin/bash
# Script for setting up monitoring with Prometheus, Grafana, and Jaeger
# This now runs BEFORE deploying the application components
set -e  # Exit immediately if any command fails

# Source utility functions
source ./setup_utils.sh

echo -e "${YELLOW}=== Setting up monitoring for Vietnamese ASR on Kubernetes ===${NC}"

# Check prerequisites
if ! command -v helm &> /dev/null; then
    echo -e "${RED}Error: helm is not installed. Please install it first.${NC}"
    echo "See: https://helm.sh/docs/intro/install/"
    exit 1
fi

# Check kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Error: kubectl is not configured or cannot connect to the cluster.${NC}"
    echo -e "Please run ./2_configure_kubernetes.sh first."
    exit 1
fi

# Check Digital Ocean load balancer limits
echo -e "\n${YELLOW}=== Checking Digital Ocean resources ===${NC}"
echo -e "Note: Digital Ocean Free Tier has a limit of 2 Load Balancers."
echo -e "We'll use NodePort for monitoring services to avoid hitting this limit."
echo -e "The main API and UI services will still use Load Balancers."

# Create required namespaces
echo -e "\n${YELLOW}=== Creating namespaces ===${NC}"
kubectl apply -f monitoring/observability-namespace.yaml
kubectl create namespace asr-system --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# Add required Helm repositories
echo -e "\n${YELLOW}=== Adding Helm repositories ===${NC}"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo update

# Install Prometheus Stack (includes Prometheus, Alertmanager, and Grafana)
echo -e "\n${YELLOW}=== Installing Prometheus Stack (this may take a few minutes) ===${NC}"
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values monitoring/prometheus-values.yaml \
  --timeout 10m0s

# Wait for Prometheus components to be ready
echo -e "\n${YELLOW}=== Waiting for Prometheus to be ready ===${NC}"
kubectl -n monitoring wait --for=condition=available --timeout=300s deployment/prometheus-kube-prometheus-operator
kubectl -n monitoring wait --for=condition=available --timeout=300s deployment/prometheus-grafana

# Install Jaeger Operator
echo -e "\n${YELLOW}=== Installing Jaeger Operator ===${NC}"
helm upgrade --install jaeger-operator jaegertracing/jaeger-operator \
  --namespace observability \
  --timeout 5m0s

# Create a Jaeger instance
echo -e "\n${YELLOW}=== Creating Jaeger instance ===${NC}"
kubectl apply -f monitoring/jaeger-instance.yaml

# Wait for Jaeger operator to be ready
echo -e "\n${YELLOW}=== Waiting for Jaeger to be ready ===${NC}"
kubectl -n observability wait --for=condition=available --timeout=300s deployment/jaeger-operator

# Get node IP for accessing NodePort services
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
if [ -z "$NODE_IP" ]; then
    # Try to get internal IP if external IP is not available
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
fi

# Get NodePort values
PROMETHEUS_PORT=$(kubectl get svc prometheus-kube-prometheus-prometheus -n monitoring -o jsonpath='{.spec.ports[0].nodePort}')
GRAFANA_PORT=$(kubectl get svc prometheus-grafana -n monitoring -o jsonpath='{.spec.ports[0].nodePort}')
JAEGER_PORT=$(kubectl get svc jaeger-query -n observability -o jsonpath='{.spec.ports[0].nodePort}')

# Display information about the deployed monitoring services
echo -e "\n${YELLOW}=== Monitoring Endpoints ===${NC}"
echo -e "Since we're using NodePort for monitoring services to stay within Digital Ocean limits,"
echo -e "you'll need to use the Node IP and port to access these services.\n"

# Show Prometheus service details
echo -e "${GREEN}Prometheus:${NC}"
kubectl get svc -n monitoring prometheus-kube-prometheus-prometheus
echo -e "Access URL: http://$NODE_IP:$PROMETHEUS_PORT"

# Show Grafana service details
echo -e "\n${GREEN}Grafana:${NC}"
kubectl get svc -n monitoring prometheus-grafana
echo -e "Access URL: http://$NODE_IP:$GRAFANA_PORT"
echo "Username: admin"
echo "Password: admin"

# Show Jaeger Query service details
echo -e "\n${GREEN}Jaeger:${NC}"
kubectl get svc -n observability jaeger-query
echo -e "Access URL: http://$NODE_IP:$JAEGER_PORT"

# Provide instructions for persistent access
echo -e "\n${YELLOW}=== For Persistent Access ===${NC}"
echo -e "For a more permanent solution, consider setting up an Ingress controller"
echo -e "to route traffic from a single LoadBalancer to multiple services."
echo -e "This would allow you to use just one LoadBalancer for all your monitoring services."

echo -e "\n${GREEN}=== Monitoring setup completed! ===${NC}"
echo "It may take a few minutes for all services to become fully available."
echo -e "\nNext step: Run ./4_deploy_application.sh to deploy the ASR application" 