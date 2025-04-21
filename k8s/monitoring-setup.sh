#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Setting up monitoring for Vietnamese ASR on Kubernetes ===${NC}"

# Check if helm is installed
if ! command -v helm &> /dev/null; then
    echo -e "${RED}Error: helm is not installed. Please install it first.${NC}"
    echo "See: https://helm.sh/docs/intro/install/"
    exit 1
fi

# Check if kubectl is configured
if ! kubectl get nodes &> /dev/null; then
    echo -e "${RED}Error: kubectl is not configured or cannot connect to the cluster.${NC}"
    exit 1
fi

# Create monitoring namespace if it doesn't exist
echo -e "\n${YELLOW}=== Creating monitoring namespace ===${NC}"
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# Add Helm repositories
echo -e "\n${YELLOW}=== Adding Helm repositories ===${NC}"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo update

# Install Prometheus Stack
echo -e "\n${YELLOW}=== Installing Prometheus Stack (this may take a few minutes) ===${NC}"
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values monitoring/prometheus-values.yaml

# Wait for Prometheus deployment
echo -e "\n${YELLOW}=== Waiting for Prometheus to be ready ===${NC}"
kubectl -n monitoring wait --for=condition=available --timeout=300s deployment/prometheus-kube-prometheus-operator
kubectl -n monitoring wait --for=condition=available --timeout=300s deployment/prometheus-grafana

# Install Jaeger Operator
echo -e "\n${YELLOW}=== Installing Jaeger Operator ===${NC}"
helm upgrade --install jaeger-operator jaegertracing/jaeger-operator \
  --namespace observability

# Apply Jaeger instance
echo -e "\n${YELLOW}=== Creating Jaeger instance ===${NC}"
kubectl apply -f monitoring/jaeger-instance.yaml

# Wait for Jaeger to be ready
echo -e "\n${YELLOW}=== Waiting for Jaeger to be ready ===${NC}"
kubectl -n observability wait --for=condition=available --timeout=300s deployment/jaeger-operator

# Get endpoints
echo -e "\n${YELLOW}=== Monitoring Endpoints ===${NC}"

echo -e "${GREEN}Prometheus:${NC}"
kubectl get svc -n monitoring prometheus-kube-prometheus-prometheus

echo -e "\n${GREEN}Grafana:${NC}"
kubectl get svc -n monitoring prometheus-grafana
echo "Username: admin"
echo "Password: admin"

echo -e "\n${GREEN}Jaeger:${NC}"
kubectl get svc -n observability jaeger-query

echo -e "\n${GREEN}=== Monitoring setup completed! ===${NC}"
echo "It may take a few minutes for all services to become fully available."
