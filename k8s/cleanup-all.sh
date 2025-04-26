#!/bin/bash
# Script to cleanup all resources deployed in the Kubernetes cluster
set -e  # Exit immediately if any command fails

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}=== Vietnamese ASR Kubernetes Cleanup Script ===${NC}"
echo -e "${RED}WARNING: This will remove ALL resources deployed by this project${NC}"
read -p "Are you sure you want to proceed? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

# Function to delete resources with namespace and wait for deletion
delete_resources() {
    local namespace=$1
    local resource_type=$2
    local selector=$3
    local timeout=${4:-60}

    if [[ -z "$selector" ]]; then
        echo -e "${YELLOW}Deleting all $resource_type in namespace $namespace...${NC}"
        kubectl delete $resource_type --all -n $namespace --ignore-not-found=true --timeout=${timeout}s
    else
        echo -e "${YELLOW}Deleting $resource_type with selector $selector in namespace $namespace...${NC}"
        kubectl delete $resource_type -l $selector -n $namespace --ignore-not-found=true --timeout=${timeout}s
    fi
}

# Function to delete a namespace and wait for its deletion
delete_namespace() {
    local namespace=$1
    
    if kubectl get namespace $namespace &>/dev/null; then
        echo -e "${YELLOW}Deleting namespace $namespace...${NC}"
        kubectl delete namespace $namespace --timeout=120s
        
        # Wait for the namespace to be deleted or timeout after 2 minutes
        local timeout=120
        local start_time=$(date +%s)
        
        echo "Waiting for namespace $namespace to be deleted..."
        while kubectl get namespace $namespace &>/dev/null; do
            local current_time=$(date +%s)
            local elapsed_time=$((current_time - start_time))
            if [[ $elapsed_time -gt $timeout ]]; then
                echo -e "${YELLOW}Timed out waiting for namespace $namespace to be deleted.${NC}"
                echo -e "${YELLOW}The namespace might still be deleting in the background.${NC}"
                break
            fi
            echo "Still waiting..."
            sleep 5
        done
    else
        echo -e "${GREEN}Namespace $namespace does not exist. Skipping.${NC}"
    fi
}

echo -e "\n${YELLOW}=== Step 1: Removing Ingress Resources ===${NC}"
# Delete ASR ingress
kubectl delete ingress asr-ingress --ignore-not-found=true

# Delete Ingress Controller and related resources
if kubectl get namespace ingress-nginx &>/dev/null; then
    echo -e "${YELLOW}Removing NGINX Ingress Controller...${NC}"
    kubectl delete -f ingress/nginx-ingress-controller.yaml --ignore-not-found=true || true
    
    # Force delete if the above didn't work
    delete_resources "ingress-nginx" "deployment" "app.kubernetes.io/name=ingress-nginx"
    delete_resources "ingress-nginx" "service" "app.kubernetes.io/name=ingress-nginx"
    delete_namespace "ingress-nginx"
else
    echo -e "${GREEN}Ingress controller namespace does not exist. Skipping.${NC}"
fi

echo -e "\n${YELLOW}=== Step 2: Removing Monitoring Components ===${NC}"
# Delete Prometheus resources
if kubectl get namespace monitoring &>/dev/null; then
    echo -e "${YELLOW}Removing Prometheus and Grafana resources...${NC}"
    delete_resources "monitoring" "deployment" "app=prometheus"
    delete_resources "monitoring" "service" "app=prometheus"
    delete_resources "monitoring" "configmap" "app=prometheus"
    
    delete_resources "monitoring" "deployment" "app=grafana"
    delete_resources "monitoring" "service" "app=grafana"
    delete_resources "monitoring" "configmap" ""
    
    # Delete all other resources in monitoring namespace
    delete_resources "monitoring" "service" ""
    delete_resources "monitoring" "deployment" ""
    delete_resources "monitoring" "statefulset" ""
    delete_resources "monitoring" "daemonset" ""
    delete_resources "monitoring" "configmap" ""
    delete_resources "monitoring" "secret" ""
    delete_resources "monitoring" "persistentvolumeclaim" ""
    
    # Delete monitoring namespace
    delete_namespace "monitoring"
else
    echo -e "${GREEN}Monitoring namespace does not exist. Skipping.${NC}"
fi

# Delete Jaeger resources
if kubectl get namespace observability &>/dev/null; then
    echo -e "${YELLOW}Removing Jaeger resources...${NC}"
    delete_resources "observability" "deployment" "app=jaeger"
    delete_resources "observability" "service" "app=jaeger"
    
    # Delete all other resources in observability namespace
    delete_resources "observability" "service" ""
    delete_resources "observability" "deployment" ""
    delete_resources "observability" "statefulset" ""
    delete_resources "observability" "daemonset" ""
    delete_resources "observability" "configmap" ""
    delete_resources "observability" "secret" ""
    
    # Delete observability namespace
    delete_namespace "observability"
else
    echo -e "${GREEN}Observability namespace does not exist. Skipping.${NC}"
fi

echo -e "\n${YELLOW}=== Step 3: Removing Application Components ===${NC}"
# Delete ASR system resources
if kubectl get namespace asr-system &>/dev/null; then
    echo -e "${YELLOW}Removing ASR API and UI components...${NC}"
    delete_resources "asr-system" "deployment" ""
    delete_resources "asr-system" "service" ""
    delete_resources "asr-system" "configmap" ""
    delete_resources "asr-system" "secret" ""
    delete_resources "asr-system" "persistentvolumeclaim" ""
    
    # Delete ASR system namespace
    delete_namespace "asr-system"
else
    echo -e "${GREEN}ASR system namespace does not exist. Skipping.${NC}"
fi

echo -e "\n${YELLOW}=== Step 4: Removing Cluster-wide Resources ===${NC}"
# Delete any cluster-level resources
echo -e "${YELLOW}Removing ClusterRoles and ClusterRoleBindings...${NC}"
kubectl delete clusterrole ingress-nginx --ignore-not-found=true
kubectl delete clusterrolebinding ingress-nginx --ignore-not-found=true

# Delete persistent volumes if there are any orphaned ones
echo -e "${YELLOW}Checking for orphaned PersistentVolumes...${NC}"
ORPHANED_PVS=$(kubectl get pv -o json | jq -r '.items[] | select(.status.phase == "Released") | .metadata.name')
if [[ ! -z "$ORPHANED_PVS" ]]; then
    echo -e "${YELLOW}Deleting orphaned PersistentVolumes:${NC}"
    echo "$ORPHANED_PVS"
    for pv in $ORPHANED_PVS; do
        kubectl delete pv $pv
    done
else
    echo -e "${GREEN}No orphaned PersistentVolumes found.${NC}"
fi

echo -e "\n${GREEN}=== Cleanup Complete! ===${NC}"
echo -e "All ASR components, monitoring tools, and Ingress resources have been removed."
echo -e "You can verify with: kubectl get all --all-namespaces" 