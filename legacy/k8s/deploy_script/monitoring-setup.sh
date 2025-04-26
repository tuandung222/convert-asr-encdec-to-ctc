#!/bin/bash
# This script sets up a comprehensive monitoring stack for the Vietnamese ASR system on Kubernetes
# It installs Prometheus, Grafana, and Jaeger for metrics, dashboards, and distributed tracing
set -e  # Exit immediately if any command fails

# Define color codes for better readability in terminal output
GREEN='\033[0;32m'   # Success messages
YELLOW='\033[1;33m'  # Section headers and warnings
RED='\033[0;31m'     # Error messages
NC='\033[0m'         # No Color (resets formatting)

echo -e "${YELLOW}=== Setting up monitoring for Vietnamese ASR on Kubernetes ===${NC}"

# Prerequisite check: Verify Helm is installed
# Helm is required to install the monitoring components as packaged charts
if ! command -v helm &> /dev/null; then
    echo -e "${RED}Error: helm is not installed. Please install it first.${NC}"
    echo "See: https://helm.sh/docs/intro/install/"
    exit 1
fi

# Prerequisite check: Verify kubectl is configured and can connect to the cluster
# This ensures we have a working connection to the Kubernetes API
if ! kubectl get nodes &> /dev/null; then
    echo -e "${RED}Error: kubectl is not configured or cannot connect to the cluster.${NC}"
    exit 1
fi

# Create a dedicated namespace for monitoring tools
# Using a separate namespace helps with organization and access control
echo -e "\n${YELLOW}=== Creating monitoring namespace ===${NC}"
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
# The --dry-run=client flag generates the manifest without applying it
# This is a safe way to create the namespace only if it doesn't exist

# Add required Helm repositories for monitoring components
echo -e "\n${YELLOW}=== Adding Helm repositories ===${NC}"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo update  # Update the local cache of available charts

# Install Prometheus Stack (includes Prometheus, Alertmanager, and Grafana)
# This provides metrics collection, alerting, and visualization capabilities
echo -e "\n${YELLOW}=== Installing Prometheus Stack (this may take a few minutes) ===${NC}"
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values monitoring/prometheus-values.yaml
# The --install flag creates a new release if it doesn't exist
# The --values flag specifies custom configuration from a local file

# Wait for Prometheus components to be ready before proceeding
# This ensures the monitoring foundation is available before adding other components
echo -e "\n${YELLOW}=== Waiting for Prometheus to be ready ===${NC}"
kubectl -n monitoring wait --for=condition=available --timeout=300s deployment/prometheus-kube-prometheus-operator
kubectl -n monitoring wait --for=condition=available --timeout=300s deployment/prometheus-grafana
# The --timeout=300s flag sets a 5-minute timeout for the wait operation

# Install Jaeger Operator for distributed tracing capabilities
# The operator pattern allows for easier management of Jaeger instances
echo -e "\n${YELLOW}=== Installing Jaeger Operator ===${NC}"
helm upgrade --install jaeger-operator jaegertracing/jaeger-operator \
  --namespace observability

# Create a Jaeger instance using the operator
# This deploys the actual Jaeger components (collector, query, agent)
echo -e "\n${YELLOW}=== Creating Jaeger instance ===${NC}"
kubectl apply -f monitoring/jaeger-instance.yaml

# Wait for Jaeger operator to be ready
# The operator will then create and manage the Jaeger instance
echo -e "\n${YELLOW}=== Waiting for Jaeger to be ready ===${NC}"
kubectl -n observability wait --for=condition=available --timeout=300s deployment/jaeger-operator

# Display information about the deployed monitoring services
# This helps users access the monitoring interfaces
echo -e "\n${YELLOW}=== Monitoring Endpoints ===${NC}"

# Show Prometheus service details (for metrics API access)
echo -e "${GREEN}Prometheus:${NC}"
kubectl get svc -n monitoring prometheus-kube-prometheus-prometheus

# Show Grafana service details (for dashboard access)
# Also provide default login credentials
echo -e "\n${GREEN}Grafana:${NC}"
kubectl get svc -n monitoring prometheus-grafana
echo "Username: admin"  # Default Grafana username
echo "Password: admin"  # Default Grafana password

# Show Jaeger Query service details (for tracing UI access)
echo -e "\n${GREEN}Jaeger:${NC}"
kubectl get svc -n observability jaeger-query

# Final success message
echo -e "\n${GREEN}=== Monitoring setup completed! ===${NC}"
echo "It may take a few minutes for all services to become fully available."
