#!/bin/bash
# Setup Nginx Ingress Controller for monitoring access
set -e  # Exit immediately if any command fails

# Source utility functions
source ./setup_utils.sh

echo -e "${YELLOW}=== Setting up Nginx Ingress Controller for Vietnamese ASR Monitoring ===${NC}"

# Check prerequisites
if ! command -v helm &> /dev/null; then
    echo -e "${RED}Error: helm is not installed. Please install it first.${NC}"
    echo "See: https://helm.sh/docs/intro/install/"
    exit 1
fi

# Check kubectl is configured using our utility function
if ! ensure_kubeconfig; then
    exit 1
fi

# Create namespace for ingress controller
kubectl create namespace ingress-nginx --dry-run=client -o yaml | kubectl apply -f -

# Add Helm repository for Nginx Ingress Controller
echo -e "\n${YELLOW}=== Adding Helm repository for Nginx Ingress Controller ===${NC}"
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Install Nginx Ingress Controller
echo -e "\n${YELLOW}=== Installing Nginx Ingress Controller (this may take a few minutes) ===${NC}"
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --set controller.metrics.enabled=true \
  --set controller.metrics.serviceMonitor.enabled=true \
  --set controller.resources.requests.memory=128Mi \
  --set controller.resources.requests.cpu=100m \
  --set controller.resources.limits.memory=256Mi \
  --set controller.resources.limits.cpu=200m \
  --timeout 5m0s

# Wait for the Ingress Controller to be ready
echo -e "\n${YELLOW}=== Waiting for Ingress Controller to be ready ===${NC}"
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=300s

# Apply Ingress resources
echo -e "\n${YELLOW}=== Creating Ingress resources for monitoring services ===${NC}"
kubectl apply -f monitoring/monitoring-ingress.yaml

# Get the Ingress Controller external IP/hostname
echo -e "\n${YELLOW}=== Getting Ingress Controller access information ===${NC}"
RETRIES=0
MAX_RETRIES=30
while true; do
  INGRESS_IP=$(kubectl get service -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
  INGRESS_HOSTNAME=$(kubectl get service -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
  
  if [[ -n "$INGRESS_IP" || -n "$INGRESS_HOSTNAME" ]]; then
    break
  fi
  
  RETRIES=$((RETRIES+1))
  if [[ $RETRIES -ge $MAX_RETRIES ]]; then
    echo -e "${YELLOW}Timed out waiting for Ingress Controller LoadBalancer. Using node port instead.${NC}"
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
    NODE_PORT=$(kubectl get service -n ingress-nginx ingress-nginx-controller -o jsonpath='{.spec.ports[0].nodePort}')
    INGRESS_ENDPOINT="${NODE_IP}:${NODE_PORT}"
    break
  fi
  
  echo -e "${YELLOW}Waiting for Ingress Controller LoadBalancer IP/hostname... ($RETRIES/$MAX_RETRIES)${NC}"
  sleep 10
done

if [[ -n "$INGRESS_IP" ]]; then
  INGRESS_ENDPOINT="$INGRESS_IP"
elif [[ -n "$INGRESS_HOSTNAME" ]]; then
  INGRESS_ENDPOINT="$INGRESS_HOSTNAME"
fi

# Display access information
echo -e "\n${GREEN}=== Monitoring Access Information ===${NC}"
echo -e "Ingress Controller endpoint: ${INGRESS_ENDPOINT}"
echo -e "\nAccess your monitoring tools through the following URLs:"
echo -e "Prometheus: http://${INGRESS_ENDPOINT}/prometheus/"
echo -e "Grafana: http://${INGRESS_ENDPOINT}/grafana/"
echo -e "Jaeger: http://${INGRESS_ENDPOINT}/jaeger/"
echo -e "\nAuthentication:"
echo -e "Username: admin"
echo -e "Password: admin"

echo -e "\n${GREEN}=== Ingress Controller setup completed! ===${NC}"
echo -e "You can now access all monitoring tools through a single IP address using path-based routing." 