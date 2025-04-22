# #!/bin/bash
# # Script for setting up monitoring with Prometheus, Grafana, and Jaeger
# set -e  # Exit immediately if any command fails

# # Source utility functions
# source ./setup_utils.sh

# echo -e "${YELLOW}=== Setting up monitoring for Vietnamese ASR on Kubernetes ===${NC}"

# # Check prerequisites
# if ! command -v helm &> /dev/null; then
#     echo -e "${RED}Error: helm is not installed. Please install it first.${NC}"
#     echo "See: https://helm.sh/docs/intro/install/"
#     exit 1
# fi

# # Check kubectl is configured
# if ! kubectl cluster-info &> /dev/null; then
#     echo -e "${RED}Error: kubectl is not configured or cannot connect to the cluster.${NC}"
#     echo -e "Please run ./2_configure_kubernetes.sh first."
#     exit 1
# fi

# # Create a dedicated namespace for monitoring tools if it doesn't exist
# echo -e "\n${YELLOW}=== Creating monitoring namespace ===${NC}"
# kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# # Add required Helm repositories
# echo -e "\n${YELLOW}=== Adding Helm repositories ===${NC}"
# helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
# helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
# helm repo update

# # Install Prometheus Stack (includes Prometheus, Alertmanager, and Grafana)
# echo -e "\n${YELLOW}=== Installing Prometheus Stack (this may take a few minutes) ===${NC}"
# helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
#   --namespace monitoring \
#   --values monitoring/prometheus-values.yaml

# # Wait for Prometheus components to be ready
# echo -e "\n${YELLOW}=== Waiting for Prometheus to be ready ===${NC}"
# kubectl -n monitoring wait --for=condition=available --timeout=300s deployment/prometheus-kube-prometheus-operator
# kubectl -n monitoring wait --for=condition=available --timeout=300s deployment/prometheus-grafana

# # Install Jaeger Operator
# echo -e "\n${YELLOW}=== Installing Jaeger Operator ===${NC}"
# helm upgrade --install jaeger-operator jaegertracing/jaeger-operator \
#   --namespace observability

# # Create a Jaeger instance
# echo -e "\n${YELLOW}=== Creating Jaeger instance ===${NC}"
# kubectl apply -f monitoring/jaeger-instance.yaml

# # Wait for Jaeger operator to be ready
# echo -e "\n${YELLOW}=== Waiting for Jaeger to be ready ===${NC}"
# kubectl -n observability wait --for=condition=available --timeout=300s deployment/jaeger-operator

# # Display information about the deployed monitoring services
# echo -e "\n${YELLOW}=== Monitoring Endpoints ===${NC}"

# # Show Prometheus service details
# echo -e "${GREEN}Prometheus:${NC}"
# kubectl get svc -n monitoring prometheus-kube-prometheus-prometheus

# # Show Grafana service details
# echo -e "\n${GREEN}Grafana:${NC}"
# kubectl get svc -n monitoring prometheus-grafana
# echo "Username: admin"
# echo "Password: admin"

# # Show Jaeger Query service details
# echo -e "\n${GREEN}Jaeger:${NC}"
# kubectl get svc -n observability jaeger-query

# echo -e "\n${GREEN}=== Monitoring setup completed! ===${NC}"
# echo "It may take a few minutes for all services to become fully available." 