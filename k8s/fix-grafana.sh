#!/bin/bash
# Script to completely reset and fix Grafana deployment
set -e

# Source utility functions 
source ./setup_utils.sh

echo -e "${YELLOW}=== Completely resetting Grafana deployment ===${NC}"

# Check if kubectl is configured using our utility function
if ! ensure_kubeconfig; then
    exit 1
fi

# Check if Grafana namespace exists
if ! kubectl get namespace monitoring &> /dev/null; then
    echo -e "${RED}Error: Monitoring namespace not found. Please run ./3_setup_monitoring.sh first.${NC}"
    exit 1
fi

echo -e "${YELLOW}=== Current Grafana status ===${NC}"
kubectl -n monitoring get pods | grep grafana || echo "No Grafana pods found"

echo -e "\n${YELLOW}=== Deleting existing Grafana resources ===${NC}"

# Delete the Grafana deployment
echo "Deleting Grafana deployment..."
kubectl -n monitoring delete deployment prometheus-grafana --ignore-not-found=true

# Delete any Grafana pods
echo "Deleting any lingering Grafana pods..."
GRAFANA_PODS=$(kubectl -n monitoring get pods | grep grafana | awk '{print $1}')
if [ ! -z "$GRAFANA_PODS" ]; then
    for pod in $GRAFANA_PODS; do
        kubectl -n monitoring delete pod $pod --grace-period=0 --force
    done
fi

# Delete PVCs
echo "Deleting Grafana PVCs..."
kubectl -n monitoring delete pvc -l app.kubernetes.io/name=grafana --ignore-not-found=true

# Wait for resources to be fully deleted
echo "Waiting for resources to be fully deleted..."
sleep 5

echo -e "\n${YELLOW}=== Applying simplified Grafana configuration ===${NC}"

# Create a simplified Grafana configuration
cat > grafana-patch.yaml << EOF
grafana:
  persistence:
    enabled: false
  initChownData:
    enabled: false
  securityContext:
    runAsUser: 0
    runAsGroup: 0
    fsGroup: 0
  adminPassword: admin
  resources:
    limits:
      cpu: 200m
      memory: 300Mi
    requests:
      cpu: 50m
      memory: 100Mi
EOF

# Apply the simplified configuration
echo "Upgrading Prometheus stack with simplified Grafana config..."
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
    -n monitoring \
    --values monitoring/prometheus-values.yaml \
    --values grafana-patch.yaml \
    --timeout 5m

# Wait for the deployment to be ready
echo -e "\n${YELLOW}=== Waiting for Grafana deployment to be ready ===${NC}"
sleep 10
kubectl -n monitoring get pods | grep grafana

# Wait for up to 2 minutes for Grafana to be ready
echo "Waiting for Grafana to be ready (up to 2 minutes)..."
for i in {1..12}; do
    if kubectl -n monitoring get pods | grep grafana | grep "Running" > /dev/null; then
        echo -e "${GREEN}Grafana is now running!${NC}"
        break
    fi
    echo "Still waiting... ($i/12)"
    sleep 10
    kubectl -n monitoring get pods | grep grafana
done

# Get Grafana access information
echo -e "\n${YELLOW}=== Grafana Access Information ===${NC}"
GRAFANA_SERVICE=$(kubectl -n monitoring get svc | grep grafana | head -1 | awk '{print $1}')
if [ ! -z "$GRAFANA_SERVICE" ]; then
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
    if [ -z "$NODE_IP" ]; then
        NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
    fi
    
    GRAFANA_PORT=$(kubectl get svc $GRAFANA_SERVICE -n monitoring -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "unknown")
    
    echo -e "${GREEN}Service:${NC} $GRAFANA_SERVICE"
    echo -e "${GREEN}URL:${NC} http://$NODE_IP:$GRAFANA_PORT"
    echo -e "${GREEN}Username:${NC} admin"
    echo -e "${GREEN}Password:${NC} admin"
else
    echo -e "${RED}Grafana service not found.${NC}"
fi

echo -e "\n${GREEN}=== Grafana reset completed! ===${NC}"
echo -e "If Grafana is still not running, try running this script again after a few minutes."
echo -e "You can also check the logs for more details:"
echo -e "kubectl -n monitoring logs \$(kubectl -n monitoring get pods -l app.kubernetes.io/name=grafana -o name | head -1)" 