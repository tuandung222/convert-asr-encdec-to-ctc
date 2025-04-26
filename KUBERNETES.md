# Vietnamese ASR Kubernetes Deployment

This document describes how to deploy the Vietnamese ASR system on Digital Ocean Kubernetes Service (DOKS).

## Architecture Overview

The Kubernetes deployment consists of:

1. **API Service (3 replicas)** - FastAPI backend for speech recognition
2. **UI Service** - Streamlit UI for interacting with the API
3. **Monitoring Stack**:
   - Prometheus for metrics collection
   - Grafana for visualization
   - Jaeger for distributed tracing
4. **CI/CD Pipeline** with Jenkins

<!-- ![Architecture Diagram](https://mermaid.ink/img/pako:eNqFk01v2zAMhv8KoVMKtECzpsPQHFI0LTDkEiCAsx52KGTRMRFZUiU6qVfkv49ynDrN2vawg0i-j0hRpFcsVgpZwPJVbJRXpM2LhI2UBlSKg5MZPYLMtqAkiVp52IEOOZN6kSJWzxJPk3hX1dqpqjWQxSVOx9BkRa4MwVT4lqB3J1_-QKyFwi1iA61LWpVt-OQjnmLLa4KG0WDQzCeV8CrXVa4qtUYlKb3DXfwk9_QyuzEWBD81JwTvSE7zORZ3CpVe7OEVdJr-BdTVVrshpjBbS6Oq2HhIXuHe3Y5p2Cx09cNxPtjMX4Tc-l7yRnKvFBXsEZNGNjWvpfm0g6_m-bHVUP-2-Piu8Wd1mCQB2yJMQ0yBxF00KQx9Xm8-cIjpMkzOHI-QGo89gZsKz5Brqs5F03PWCfwbKUOuoHBwJ9ViD7XYKs84b8Xm0gpvLcLsP27lDLy0WmFR_yozj6CuylTYKd3cOPa-fBiF7_1wGl4V9F1Hv4CwnftS0W9aujPJSjgzs2AVecSNTJWsYV4Ym5WJLECVojPGlVQoO5M6baiQB-QY1HbSVK3r_3Y5Lw0-nRs2hR_KpMZy2YwbZPo4HVuaHuJMqtrSX2MpO2o32i0LWAmdqJRFY9v3HgsG8fQ0fX6OomnQXQ2HQS8KetPpaNC_Daa9YLrqB4PwKgpXw96oPwwHqwDeQb_YPwKQVnQ?type=png) -->

## Project Structure

```
deploy/
├── k8s/                  # Kubernetes manifests
│   ├── base/             # Base manifests for API and UI
│   │   ├── namespace.yaml
│   │   ├── api-deployment.yaml
│   │   ├── api-service.yaml
│   │   ├── ui-deployment.yaml
│   │   └── ui-service.yaml
│   ├── monitoring/       # Monitoring configuration
│   │   ├── prometheus-values.yaml
│   │   ├── jaeger-instance.yaml
│   │   └── observability-namespace.yaml
│   ├── setup.sh          # Main setup script
│   └── monitoring-setup.sh  # Monitoring setup script
├── terraform/            # Infrastructure as Code
│   ├── main.tf           # Main Terraform configuration
│   ├── variables.tf      # Input variables
│   ├── outputs.tf        # Output values
│   └── terraform.tfvars.example  # Example variables file
└── Jenkinsfile           # CI/CD pipeline definition
```

## Deployment Steps

### 1. Prerequisites

- [Digital Ocean](https://www.digitalocean.com/) account with API token
- [Terraform](https://www.terraform.io/) (v1.0+)
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- [Helm](https://helm.sh/) (v3.0+)
- [doctl](https://docs.digitalocean.com/reference/doctl/) - Digital Ocean CLI

### 2. Automated Deployment

The easiest way to deploy is using the provided setup script:

```bash
# Navigate to the k8s directory
cd k8s

# Make scripts executable (on Linux/Mac)
chmod +x setup.sh monitoring-setup.sh

# Run the setup script
./setup.sh
```

This script will:
1. Create a Kubernetes cluster on Digital Ocean
2. Deploy the ASR API with 3 replicas
3. Deploy the UI component
4. Optionally set up the monitoring stack

### 3. Manual Deployment

For more control, follow these steps:

1. **Create infrastructure with Terraform**:
   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your DO API token
   terraform init
   terraform apply
   ```

2. **Configure kubectl**:
   ```bash
   doctl kubernetes cluster kubeconfig save asr-k8s-cluster
   ```

3. **Deploy application**:
   ```bash
   kubectl apply -f k8s/monitoring/observability-namespace.yaml
   kubectl apply -f k8s/base/namespace.yaml
   kubectl apply -f k8s/base/
   ```

4. **Set up monitoring**:
   ```bash
   cd k8s
   ./monitoring-setup.sh
   ```

## Access the Application

After deployment, get the service IP addresses:

```bash
kubectl get svc -n asr-system
```

Access the services:
- ASR API: http://<api-loadbalancer-ip>
- ASR UI: http://<ui-loadbalancer-ip>
- Grafana: http://<grafana-loadbalancer-ip> (Username: admin, Password: admin)
- Jaeger UI: http://<jaeger-query-loadbalancer-ip>:16686

## CI/CD with Jenkins

The included Jenkinsfile automates the build and deployment process:

1. Builds Docker images
2. Pushes images to Docker Hub
3. Deploys to Kubernetes
4. Sets up monitoring

To use it:
1. Set up Jenkins with the required plugins
2. Add credentials for Docker Hub and Digital Ocean
3. Create a pipeline job pointing to your repository

## Troubleshooting

### Common Issues

1. **Pods in Pending state**:
   ```bash
   kubectl get pods -n asr-system
   kubectl describe pod <pod-name> -n asr-system
   ```
   Usually caused by insufficient resources or incorrect node selectors.

2. **LoadBalancer issues**:
   ```bash
   kubectl get svc -n asr-system
   ```
   If EXTERNAL-IP is <pending>, wait a bit longer or check Digital Ocean LoadBalancer status.

3. **Monitoring not working**:
   ```bash
   kubectl get pods -n monitoring
   kubectl logs <prometheus-pod> -n monitoring
   ```

## Cleaning Up

To delete all resources:

```bash
terraform destroy
```

This will delete the Kubernetes cluster and all associated resources.

## Monitoring Stack

The deployment includes a monitoring stack with Prometheus, Grafana, and Jaeger, deployed directly as Kubernetes manifests without requiring Helm:

### Components

1. **Prometheus** - Collects metrics from the ASR system
   - Uses emptyDir volume for ephemeral storage
   - Configured with NodePort service for access
   - Includes scraping configurations for ASR API and Kubernetes nodes

2. **Grafana** - Visualizes the collected metrics
   - Pre-configured with Prometheus datasource
   - Includes a basic ASR system dashboard
   - Default credentials: admin/admin
   - Uses NodePort service for access

3. **Jaeger** - Distributed tracing for API requests
   - All-in-one deployment (suitable for non-production use)
   - Query interface available via NodePort
   - Collector services available to the ASR API

### Deployment

The monitoring stack can be deployed in two ways:

1. **As part of the main deployment**: The monitoring stack is deployed automatically by the `3_setup_monitoring.sh` script before the application components.

2. **Standalone deployment**: You can deploy only the monitoring components:
   ```bash
   cd k8s/monitoring
   ./deploy-monitoring.sh
   ```

### Accessing the Monitoring Stack

After deployment, access the monitoring components using the NodeIP and NodePort:

```bash
# Get the NodeIP
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')

# Get the NodePorts
PROM_PORT=$(kubectl get svc prometheus -n monitoring -o jsonpath='{.spec.ports[0].nodePort}')
GRAFANA_PORT=$(kubectl get svc grafana -n monitoring -o jsonpath='{.spec.ports[0].nodePort}')
JAEGER_PORT=$(kubectl get svc jaeger-query -n observability -o jsonpath='{.spec.ports[0].nodePort}')

# Access URLs
echo "Prometheus: http://$NODE_IP:$PROM_PORT"
echo "Grafana: http://$NODE_IP:$GRAFANA_PORT (admin/admin)"
echo "Jaeger UI: http://$NODE_IP:$JAEGER_PORT"
```

### Resource Usage

The monitoring stack is configured with moderate resource requests and limits:

- **Prometheus**: 200m-500m CPU, 500Mi-1Gi Memory
- **Grafana**: 100m-500m CPU, 256Mi-512Mi Memory
- **Jaeger**: 100m-500m CPU, 256Mi-512Mi Memory

These values can be adjusted in the respective deployment YAML files as needed for your environment.

## Ingress Controller

To simplify access to services and reduce the number of required LoadBalancers, the deployment provides an NGINX Ingress Controller setup:

### Benefits

- Single LoadBalancer for all services (saving costs)
- Path-based routing to all services
- Simplified access through a single IP address
- Ability to add TLS/SSL termination in one place

### Architecture

```
                            ┌─────────────┐
                            │ LoadBalancer│
                            │  (1 IP)     │
                            └──────┬──────┘
                                   │
                          ┌────────▼───────┐
                          │ NGINX Ingress  │
                          │  Controller    │
                          └┬──────┬───────┬┘
                           │      │       │
      ┌────────────────────┼──────┼───────┼────────────────────┐
      │                    │      │       │                    │
┌─────▼─────┐      ┌───────▼────┐ │ ┌─────▼─────┐      ┌───────▼────┐
│  ASR API   │      │   ASR UI   │ │ │ Prometheus│      │   Jaeger   │
│ (/api/...) │      │ (/ui/...)  │ │ │ (/prom/...)│      │ (/jaeger/)│
└───────────┘      └────────────┘ │ └───────────┘      └────────────┘
                                  │
                                  │
                            ┌─────▼─────┐
                            │  Grafana  │
                            │ (/grafana)│
                            └───────────┘
```

### Path Routing

The Ingress controller routes requests based on URL paths:

- `/api/*` → ASR API service
- `/ui/*` → ASR UI (Streamlit)
- `/prometheus/*` → Prometheus metrics
- `/grafana/*` → Grafana dashboards
- `/jaeger/*` → Jaeger tracing UI
- `/` → Root path (redirects to UI)

### Setup

The Ingress Controller is set up with the following script, which should be run after deploying the application:

```bash
# From the k8s directory
cd ingress
./setup-ingress.sh
```

This script:
1. Deploys the NGINX Ingress Controller
2. Updates all services to use ClusterIP instead of LoadBalancer/NodePort
3. Creates the Ingress resource to route traffic
4. Waits for the LoadBalancer IP to be assigned
5. Displays access URLs for all services

### Access

After setup, you can access all services through a single IP address:

```
http://<ingress-ip>/api      # ASR API
http://<ingress-ip>/ui       # ASR UI
http://<ingress-ip>/prometheus  # Prometheus 
http://<ingress-ip>/grafana     # Grafana (admin/admin)
http://<ingress-ip>/jaeger      # Jaeger UI
http://<ingress-ip>/            # Root path (redirects to UI)
```
