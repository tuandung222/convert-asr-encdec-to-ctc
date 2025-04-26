#!/bin/bash
# Script to deploy the monitoring stack for Vietnamese ASR

set -e  # Exit immediately if any command fails

echo "=== Deploying Monitoring Stack for Vietnamese ASR ==="

echo "Creating namespaces..."
kubectl apply -f namespace.yaml

echo "Deploying Prometheus..."
kubectl apply -f prometheus-configmap.yaml
kubectl apply -f prometheus-deployment.yaml
kubectl apply -f prometheus-service.yaml

echo "Deploying Grafana..."
kubectl apply -f grafana-configmap.yaml
kubectl apply -f grafana-deployment.yaml
kubectl apply -f grafana-service.yaml

echo "Deploying Jaeger..."
kubectl apply -f jaeger-instance.yaml

echo "Waiting for deployments to be ready..."
kubectl -n monitoring rollout status deployment prometheus --timeout=300s || true
kubectl -n monitoring rollout status deployment grafana --timeout=180s || true
kubectl -n observability rollout status deployment jaeger --timeout=180s || true

echo "Getting service access information..."
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
if [ -z "$NODE_IP" ]; then
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
fi

PROM_PORT=$(kubectl get svc prometheus -n monitoring -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "unknown")
GRAFANA_PORT=$(kubectl get svc grafana -n monitoring -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "unknown")
JAEGER_PORT=$(kubectl get svc jaeger-query -n observability -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "unknown")

echo "=== Monitoring Endpoints ==="
echo "Prometheus: http://$NODE_IP:$PROM_PORT"
echo "Grafana: http://$NODE_IP:$GRAFANA_PORT (admin/admin)"
echo "Jaeger UI: http://$NODE_IP:$JAEGER_PORT"

echo "=== Monitoring stack deployed successfully! ===" 