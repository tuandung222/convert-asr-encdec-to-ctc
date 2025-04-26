#!/bin/bash
# Script for setting up NGINX Ingress Controller to access all services
set -e

# Source utility functions if available
[[ -f "../setup_utils.sh" ]] && source ../setup_utils.sh
[[ -f "./setup_utils.sh" ]] && source ./setup_utils.sh

# Define colors for output
GREEN=${GREEN:-'\033[0;32m'}
YELLOW=${YELLOW:-'\033[1;33m'}
RED=${RED:-'\033[0;31m'}
NC=${NC:-'\033[0m'}

echo -e "${YELLOW}=== Setting up NGINX Ingress Controller for Vietnamese ASR ===${NC}"

# Check prerequisites
check_prerequisites() {
  if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed. Please install it first.${NC}"
    exit 1
  fi

  if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Error: Cannot connect to the Kubernetes cluster.${NC}"
    echo "Please ensure that:"
    echo "1. You have a Kubernetes cluster running"
    echo "2. Your kubectl is properly configured to connect to the cluster"
    exit 1
  fi
}

# Clean up previous installation
cleanup_ingress() {
  if kubectl get namespace ingress-nginx &> /dev/null; then
    echo -e "${YELLOW}Ingress-nginx namespace already exists. Cleaning up previous installation...${NC}"
    kubectl delete -f nginx-ingress-controller.yaml --ignore-not-found=true
    sleep 10
  fi
}

# Deploy NGINX Ingress Controller
deploy_ingress_controller() {
  echo -e "\n${YELLOW}=== Deploying NGINX Ingress Controller ===${NC}"
  kubectl apply -f nginx-ingress-controller.yaml
  
  echo -e "\n${YELLOW}Waiting for ingress-nginx pod to be created...${NC}"
  sleep 5
  
  echo -e "\n${YELLOW}Pod status:${NC}"
  kubectl get pods -n ingress-nginx
}

# Update service types to ClusterIP
update_service_types() {
  echo -e "\n${YELLOW}=== Updating service types to ClusterIP for existing services ===${NC}"
  
  # Check for monitoring namespace and services
  if kubectl get namespace monitoring &> /dev/null; then
    # Update Prometheus service
    kubectl get svc -n monitoring prometheus &> /dev/null && \
      echo "Updating Prometheus service to ClusterIP..." && \
      kubectl patch svc -n monitoring prometheus -p '{"spec": {"type": "ClusterIP"}}'
    
    # Update Grafana service
    kubectl get svc -n monitoring grafana &> /dev/null && \
      echo "Updating Grafana service to ClusterIP..." && \
      kubectl patch svc -n monitoring grafana -p '{"spec": {"type": "ClusterIP"}}'
  else
    echo -e "${YELLOW}Monitoring namespace does not exist. Skipping monitoring services.${NC}"
  fi
  
  # Check for observability namespace and Jaeger service
  if kubectl get namespace observability &> /dev/null; then
    kubectl get svc -n observability jaeger-query &> /dev/null && \
      echo "Updating Jaeger Query service to ClusterIP..." && \
      kubectl patch svc -n observability jaeger-query -p '{"spec": {"type": "ClusterIP"}}'
  else
    echo -e "${YELLOW}Observability namespace does not exist. Skipping Jaeger services.${NC}"
  fi
  
  # Check for ASR system namespace and services
  if kubectl get namespace asr-system &> /dev/null; then
    # Update ASR API service
    kubectl get svc -n asr-system asr-api &> /dev/null && \
      echo "Updating ASR API service to ClusterIP..." && \
      kubectl patch svc -n asr-system asr-api -p '{"spec": {"type": "ClusterIP"}}'
    
    # Update ASR UI service
    kubectl get svc -n asr-system asr-ui &> /dev/null && \
      echo "Updating ASR UI service to ClusterIP..." && \
      kubectl patch svc -n asr-system asr-ui -p '{"spec": {"type": "ClusterIP"}}'
  else
    echo -e "${YELLOW}ASR-system namespace does not exist. Skipping ASR services.${NC}"
  fi
}

# Wait for the NGINX Ingress Controller to be ready
wait_for_controller() {
  echo -e "\n${YELLOW}=== Waiting for NGINX Ingress Controller to be ready ===${NC}"
  echo -e "${YELLOW}Note: This may take a few minutes for the first deployment...${NC}"
  
  # Get the pod name for checking logs later if needed
  CONTROLLER_POD=$(kubectl get pod -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
  
  # Try to wait for the deployment to be ready
  if ! kubectl -n ingress-nginx rollout status deployment nginx-ingress-controller --timeout=180s; then
    echo -e "${YELLOW}Timeout waiting for controller. Checking status and continuing anyway...${NC}"
    kubectl get pods -n ingress-nginx
    
    if [ -n "$CONTROLLER_POD" ]; then
      echo -e "${YELLOW}Checking controller pod logs:${NC}"
      kubectl logs -n ingress-nginx $CONTROLLER_POD --tail=20
    fi
    
    echo -e "${YELLOW}Proceeding with ingress setup despite timeout...${NC}"
  fi
}

# Create and apply the Ingress configuration
create_ingress_config() {
  local temp_file=$(mktemp)
  echo -e "\n${YELLOW}=== Creating dynamic Ingress resource based on available services ===${NC}"
  
  # Create header of ingress configuration
  cat > $temp_file << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: asr-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /\$2
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      proxy_set_header X-Forwarded-Proto \$scheme;
      proxy_set_header X-Forwarded-Host \$host;
      proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
spec:
  ingressClassName: nginx
  rules:
  - http:
      paths:
EOF

  # Add ASR services if they exist
  if kubectl get namespace asr-system &> /dev/null; then
    # Add ASR API
    if kubectl get svc -n asr-system asr-api &> /dev/null; then
      cat >> $temp_file << EOF
      # ASR API - Main API endpoint
      - path: /api(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: asr-api
            port:
              number: 8000
EOF
      echo "Added ASR API to Ingress"
    fi
    
    # Add ASR UI
    if kubectl get svc -n asr-system asr-ui &> /dev/null; then
      cat >> $temp_file << EOF
      # ASR UI - Streamlit UI
      - path: /ui(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: asr-ui
            port:
              number: 8501
      
      # Root path redirect to UI
      - path: /
        pathType: Exact
        backend:
          service:
            name: asr-ui
            port:
              number: 8501
EOF
      echo "Added ASR UI to Ingress"
    fi
  fi

  # Add monitoring services if they exist
  if kubectl get namespace monitoring &> /dev/null; then
    # Add Prometheus
    if kubectl get svc -n monitoring prometheus &> /dev/null; then
      cat >> $temp_file << EOF
      # Prometheus - Metrics
      - path: /prometheus(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: prometheus
            port:
              number: 9090
EOF
      echo "Added Prometheus to Ingress"
    fi
    
    # Add Grafana
    if kubectl get svc -n monitoring grafana &> /dev/null; then
      cat >> $temp_file << EOF
      # Grafana - Dashboards
      - path: /grafana(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: grafana
            port:
              number: 3000
EOF
      echo "Added Grafana to Ingress"
    fi
  fi

  # Add Jaeger if it exists
  if kubectl get namespace observability &> /dev/null; then
    if kubectl get svc -n observability jaeger-query &> /dev/null; then
      cat >> $temp_file << EOF
      # Jaeger - Tracing UI
      - path: /jaeger(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: jaeger-query
            port:
              number: 16686
EOF
      echo "Added Jaeger to Ingress"
    fi
  fi

  # Add placeholder if no services were found
  if ! kubectl get namespace asr-system &> /dev/null && \
     ! kubectl get namespace monitoring &> /dev/null && \
     ! kubectl get namespace observability &> /dev/null; then
    echo -e "${YELLOW}No services were found. Creating a placeholder Ingress.${NC}"
    cat >> $temp_file << EOF
      # Placeholder path - replace later when services are deployed
      - path: /
        pathType: Exact
        backend:
          service:
            name: default-http-backend
            port:
              number: 80
EOF
  fi

  # Apply the configuration
  echo -e "\n${YELLOW}=== Applying dynamic Ingress configuration ===${NC}"
  kubectl apply -f $temp_file || true
  rm $temp_file
}

# Wait for and report LoadBalancer IP
get_loadbalancer_ip() {
  echo -e "\n${YELLOW}=== Getting the Ingress LoadBalancer IP ===${NC}"
  echo "Waiting for the Ingress LoadBalancer IP to be assigned..."
  echo "This may take up to 2 minutes..."

  timeout=120
  start_time=$(date +%s)
  external_ip=""

  while [ -z "$external_ip" ]; do
    echo "Waiting for LoadBalancer IP..."
    external_ip=$(kubectl get svc ingress-nginx -n ingress-nginx --template="{{range .status.loadBalancer.ingress}}{{.ip}}{{end}}" 2>/dev/null || echo "")
    
    # Check for timeout
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [ $elapsed_time -gt $timeout ]; then
      echo -e "${YELLOW}Timed out waiting for LoadBalancer IP.${NC}"
      echo -e "${YELLOW}Service status:${NC}"
      kubectl get svc -n ingress-nginx
      echo -e "${YELLOW}You may need to check your cloud provider dashboard for the LoadBalancer IP.${NC}"
      break
    fi
    
    [ -z "$external_ip" ] && sleep 10
  done

  # Display the results
  if [ -n "$external_ip" ]; then
    display_access_urls "$external_ip"
  else
    echo -e "\n${YELLOW}=== Unable to get LoadBalancer IP automatically ===${NC}"
    echo -e "Get the IP manually when it becomes available with:"
    echo -e "kubectl get svc -n ingress-nginx ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}'"
  fi
}

# Display access URLs
display_access_urls() {
  local ip=$1
  echo -e "\n${GREEN}=== Ingress LoadBalancer IP: $ip ===${NC}"
  echo -e "Once services are deployed, you can access them at the following URLs:"
  
  if kubectl get namespace asr-system &> /dev/null; then
    kubectl get svc -n asr-system asr-api &> /dev/null && echo -e "ASR API:      http://$ip/api"
    kubectl get svc -n asr-system asr-ui &> /dev/null && echo -e "ASR UI:       http://$ip/ui\nDefault Path: http://$ip/ (redirects to UI)"
  fi
  
  if kubectl get namespace monitoring &> /dev/null; then
    kubectl get svc -n monitoring prometheus &> /dev/null && echo -e "Prometheus:   http://$ip/prometheus"
    kubectl get svc -n monitoring grafana &> /dev/null && echo -e "Grafana:      http://$ip/grafana (admin/admin)"
  fi
  
  if kubectl get namespace observability &> /dev/null; then
    kubectl get svc -n observability jaeger-query &> /dev/null && echo -e "Jaeger:       http://$ip/jaeger"
  fi
}

# Show next steps if services are missing
show_next_steps() {
  if ! kubectl get namespace asr-system &> /dev/null || \
     ! kubectl get namespace monitoring &> /dev/null || \
     ! kubectl get namespace observability &> /dev/null; then
    echo -e "\n${YELLOW}=== Next Steps ===${NC}"
    echo -e "Some services have not yet been deployed. Run these scripts to deploy them:"
    
    ! kubectl get namespace monitoring &> /dev/null && echo -e "  ./3_setup_monitoring.sh     # Deploy Prometheus, Grafana, and Jaeger"
    ! kubectl get namespace asr-system &> /dev/null && echo -e "  ./4_deploy_application.sh   # Deploy ASR API and UI"
    
    echo -e "\nAfter deploying these components, run this script again to update the Ingress configuration."
  fi
}

# Main execution
main() {
  # Ensure we're in the correct directory
  if [[ ! -f "nginx-ingress-controller.yaml" ]]; then
    if [[ -d "ingress" ]]; then
      cd ingress
    else
      echo -e "${RED}Error: Could not find nginx-ingress-controller.yaml. Make sure you're in the right directory.${NC}"
      exit 1
    fi
  fi

  check_prerequisites
  cleanup_ingress
  deploy_ingress_controller
#   update_service_types
  wait_for_controller
  create_ingress_config
  get_loadbalancer_ip
  show_next_steps
  
  echo -e "\n${GREEN}=== Ingress setup completed! ===${NC}"
}

# Execute main function
main 