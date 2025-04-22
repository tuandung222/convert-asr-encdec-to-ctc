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

# Check kubectl is configured using our utility function
if ! ensure_kubeconfig; then
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
echo -e "${YELLOW}Getting actual deployment names...${NC}"

# Get actual deployment names using label selectors
PROM_OPERATOR_DEPLOYMENT=$(kubectl -n monitoring get deployments -l app.kubernetes.io/name=kube-prometheus-stack-operator -o name 2>/dev/null || echo "")
GRAFANA_DEPLOYMENT=$(kubectl -n monitoring get deployments -l app.kubernetes.io/name=grafana -o name 2>/dev/null || echo "")

if [ -z "$PROM_OPERATOR_DEPLOYMENT" ]; then
    echo -e "${YELLOW}Prometheus operator deployment not found with expected labels, trying alternative...${NC}"
    PROM_OPERATOR_DEPLOYMENT=$(kubectl -n monitoring get deployments -l app=kube-prometheus-stack-operator -o name 2>/dev/null || echo "")
    if [ -z "$PROM_OPERATOR_DEPLOYMENT" ]; then
        echo -e "${YELLOW}Listing all deployments in monitoring namespace for debugging:${NC}"
        kubectl -n monitoring get deployments
        echo -e "${YELLOW}Continuing without waiting for operator...${NC}"
    fi
fi

if [ -z "$GRAFANA_DEPLOYMENT" ]; then
    echo -e "${YELLOW}Grafana deployment not found with expected labels, trying alternative...${NC}"
    GRAFANA_DEPLOYMENT=$(kubectl -n monitoring get deployments -l app=grafana -o name 2>/dev/null || echo "")
    if [ -z "$GRAFANA_DEPLOYMENT" ]; then
        echo -e "${YELLOW}Continuing without waiting for Grafana...${NC}"
    fi
fi

# Wait for components if they were found
if [ ! -z "$PROM_OPERATOR_DEPLOYMENT" ]; then
    echo -e "${YELLOW}Waiting for $PROM_OPERATOR_DEPLOYMENT to be ready...${NC}"
    kubectl -n monitoring wait --for=condition=available --timeout=300s $PROM_OPERATOR_DEPLOYMENT
else
    # Sleep to allow time for resources to be created
    echo -e "${YELLOW}Sleeping for 30 seconds to allow Prometheus components to start...${NC}"
    sleep 30
fi

if [ ! -z "$GRAFANA_DEPLOYMENT" ]; then
    echo -e "${YELLOW}Waiting for $GRAFANA_DEPLOYMENT to be ready...${NC}"
    if ! kubectl -n monitoring wait --for=condition=available --timeout=180s $GRAFANA_DEPLOYMENT; then
        echo -e "${YELLOW}Grafana deployment not ready after timeout. This is not critical - we'll continue.${NC}"
        echo -e "${YELLOW}=== Diagnosing Grafana issues ===${NC}"
        echo -e "You can check the Grafana pod status with:"
        echo -e "  kubectl -n monitoring get pods | grep grafana"
        echo -e "Check Grafana pod logs with:"
        echo -e "  kubectl -n monitoring logs \$(kubectl -n monitoring get pods -l app.kubernetes.io/name=grafana -o name)"
        echo -e "Check Grafana pod events with:"
        echo -e "  kubectl -n monitoring describe pod \$(kubectl -n monitoring get pods -l app.kubernetes.io/name=grafana -o name | head -1)"
    else
        echo -e "${GREEN}Grafana deployment is ready!${NC}"
    fi
fi

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

# Get services and their NodePort values
echo -e "\n${YELLOW}=== Finding Prometheus services ===${NC}"
PROM_SERVICE=$(kubectl get svc -n monitoring -l app=kube-prometheus-stack-prometheus -o name 2>/dev/null || echo "")
GRAFANA_SERVICE=$(kubectl get svc -n monitoring -l app.kubernetes.io/name=grafana -o name 2>/dev/null || echo "")
JAEGER_SERVICE=$(kubectl get svc -n observability -l app=jaeger-query -o name 2>/dev/null || echo "")

# If services weren't found with the expected labels, try alternatives
if [ -z "$PROM_SERVICE" ]; then
    echo -e "${YELLOW}Prometheus service not found with expected labels, trying alternative...${NC}"
    PROM_SERVICE=$(kubectl get svc -n monitoring | grep prometheus | grep -v alertmanager | grep -v operator | head -1 | awk '{print $1}')
    PROM_SERVICE="service/$PROM_SERVICE"
fi

if [ -z "$GRAFANA_SERVICE" ]; then
    echo -e "${YELLOW}Grafana service not found with expected labels, trying alternative...${NC}"
    GRAFANA_SERVICE=$(kubectl get svc -n monitoring | grep grafana | head -1 | awk '{print $1}')
    GRAFANA_SERVICE="service/$GRAFANA_SERVICE"
fi

if [ -z "$JAEGER_SERVICE" ]; then
    echo -e "${YELLOW}Jaeger service not found with expected labels, trying alternative...${NC}"
    JAEGER_SERVICE=$(kubectl get svc -n observability | grep jaeger-query | head -1 | awk '{print $1}')
    JAEGER_SERVICE="service/$JAEGER_SERVICE"
fi

# Extract service names from the full service paths
PROM_SERVICE_NAME=$(echo $PROM_SERVICE | sed 's|^service/||')
GRAFANA_SERVICE_NAME=$(echo $GRAFANA_SERVICE | sed 's|^service/||')
JAEGER_SERVICE_NAME=$(echo $JAEGER_SERVICE | sed 's|^service/||')

# Get NodePort values for the services (if they exist)
if [ ! -z "$PROM_SERVICE_NAME" ]; then
    PROMETHEUS_PORT=$(kubectl get svc $PROM_SERVICE_NAME -n monitoring -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "unknown")
else
    PROMETHEUS_PORT="unknown"
fi

if [ ! -z "$GRAFANA_SERVICE_NAME" ]; then
    GRAFANA_PORT=$(kubectl get svc $GRAFANA_SERVICE_NAME -n monitoring -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "unknown")
else
    GRAFANA_PORT="unknown"
fi

if [ ! -z "$JAEGER_SERVICE_NAME" ]; then
    JAEGER_PORT=$(kubectl get svc $JAEGER_SERVICE_NAME -n observability -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "unknown")
else
    JAEGER_PORT="unknown"
fi

# Display information about the deployed monitoring services
echo -e "\n${YELLOW}=== Monitoring Endpoints ===${NC}"
echo -e "Since we're using NodePort for monitoring services to stay within Digital Ocean limits,"
echo -e "you'll need to use the Node IP and port to access these services.\n"

# Check if Grafana is having issues and provide common fixes
if kubectl -n monitoring get pods | grep grafana | grep -E "0/1|Error|CrashLoopBackOff" > /dev/null; then
    echo -e "\n${YELLOW}=== Grafana Troubleshooting ===${NC}"
    echo -e "Grafana pod is not starting properly. Here are some common fixes:"
    echo -e "1. Check if it's a resource issue:"
    echo -e "   kubectl -n monitoring describe pod \$(kubectl -n monitoring get pods -l app.kubernetes.io/name=grafana -o name | head -1) | grep -A5 Events"
    
    echo -e "\n2. If it's a PersistentVolumeClaim issue, you can edit the helm values to use emptyDir instead:"
    echo -e "   Create a file grafana-patch.yaml with:"
    echo -e "   grafana:\n     persistence:\n       enabled: false"
    echo -e "   Then update the deployment:"
    echo -e "   helm upgrade prometheus prometheus-community/kube-prometheus-stack -n monitoring --values monitoring/prometheus-values.yaml --values grafana-patch.yaml"
    
    echo -e "\n3. If it's a permission issue, you can try running Grafana as root:"
    echo -e "   kubectl -n monitoring patch deployment prometheus-grafana -p '{\"spec\":{\"template\":{\"spec\":{\"securityContext\":{\"runAsUser\":0,\"runAsGroup\":0,\"fsGroup\":0}}}}}'"
    
    echo -e "\n4. Inspect Grafana logs to find the specific issue:"
    echo -e "   kubectl -n monitoring logs \$(kubectl -n monitoring get pods -l app.kubernetes.io/name=grafana -o name)"
    
    echo -e "\nYou can proceed with the deployment and fix Grafana later. The API will still work without Grafana."
fi

# Show Prometheus service details
echo -e "${GREEN}Prometheus:${NC}"
if [ ! -z "$PROM_SERVICE_NAME" ]; then
    kubectl get svc -n monitoring $PROM_SERVICE_NAME
    if [ "$PROMETHEUS_PORT" != "unknown" ]; then
        echo -e "Access URL: http://$NODE_IP:$PROMETHEUS_PORT"
    else
        echo -e "Port could not be determined. Please check the service manually."
    fi
else
    echo -e "${YELLOW}Prometheus service not found. Please check manually with: kubectl get svc -n monitoring${NC}"
fi

# Show Grafana service details
echo -e "\n${GREEN}Grafana:${NC}"
if [ ! -z "$GRAFANA_SERVICE_NAME" ]; then
    kubectl get svc -n monitoring $GRAFANA_SERVICE_NAME
    if [ "$GRAFANA_PORT" != "unknown" ]; then
        echo -e "Access URL: http://$NODE_IP:$GRAFANA_PORT"
    else
        echo -e "Port could not be determined. Please check the service manually."
    fi
    echo "Username: admin"
    
    # Get Grafana password
    echo -e "${YELLOW}Getting Grafana password...${NC}"
    GRAFANA_SECRET=$(kubectl -n monitoring get secrets -l app.kubernetes.io/name=grafana -o name 2>/dev/null || echo "")
    if [ -z "$GRAFANA_SECRET" ]; then
        echo -e "${YELLOW}Grafana secret not found with expected labels, trying alternative...${NC}"
        GRAFANA_SECRET=$(kubectl -n monitoring get secrets | grep grafana | head -1 | awk '{print $1}')
    fi
    
    if [ ! -z "$GRAFANA_SECRET" ]; then
        PASSWORD=$(kubectl --namespace monitoring get secret $GRAFANA_SECRET -o jsonpath="{.data.admin-password}" | base64 -d 2>/dev/null || echo "admin")
        echo "Password: $PASSWORD"
    else
        echo "Password: admin (default)"
    fi
else
    echo -e "${YELLOW}Grafana service not found. Please check manually with: kubectl get svc -n monitoring${NC}"
fi

# Show Jaeger Query service details
echo -e "\n${GREEN}Jaeger:${NC}"
if [ ! -z "$JAEGER_SERVICE_NAME" ]; then
    kubectl get svc -n observability $JAEGER_SERVICE_NAME
    if [ "$JAEGER_PORT" != "unknown" ]; then
        echo -e "Access URL: http://$NODE_IP:$JAEGER_PORT"
    else
        echo -e "Port could not be determined. Please check the service manually."
    fi
else
    echo -e "${YELLOW}Jaeger service not found. Please check manually with: kubectl get svc -n observability${NC}"
fi

# Provide instructions for persistent access
echo -e "\n${YELLOW}=== For Persistent Access ===${NC}"
echo -e "For a more permanent solution, consider setting up an Ingress controller"
echo -e "to route traffic from a single LoadBalancer to multiple services."
echo -e "This would allow you to use just one LoadBalancer for all your monitoring services."

echo -e "\n${GREEN}=== Monitoring setup completed! ===${NC}"
echo "It may take a few minutes for all services to become fully available."
echo -e "\nNext step: Run ./4_deploy_application.sh to deploy the ASR application" 