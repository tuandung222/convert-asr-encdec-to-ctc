# Vietnamese ASR Kubernetes Deployment

This directory contains configurations for deploying the Vietnamese ASR system on Digital Ocean Kubernetes Service (DOKS).

## Prerequisites

- [Digital Ocean](https://www.digitalocean.com/) account with API token
- [Terraform](https://www.terraform.io/) (v1.0+)
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- [Helm](https://helm.sh/) (v3.0+)
- [doctl](https://docs.digitalocean.com/reference/doctl/) - Digital Ocean CLI

## Deployment Steps

### 1. Infrastructure Provisioning with Terraform

```bash
# Navigate to the terraform directory
cd terraform

# Create a terraform.tfvars file with your Digital Ocean API token
echo 'do_token = "your-digitalocean-api-token"' > terraform.tfvars

# Initialize Terraform
terraform init

# Plan the deployment
terraform plan -out=tfplan

# Apply the configuration
terraform apply tfplan

# Get the kubeconfig
export KUBECONFIG=$(pwd)/kubeconfig.yaml
```

### 2. Deploy Application to Kubernetes

```bash
# Create namespaces
kubectl apply -f k8s/monitoring/observability-namespace.yaml
kubectl apply -f k8s/base/namespace.yaml

# Deploy the API and UI components
kubectl apply -f k8s/base/

# Verify deployments
kubectl get pods -n asr-system
```

### 3. Install Monitoring Stack

```bash
# Add Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo update

# Install Prometheus Stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --values k8s/monitoring/prometheus-values.yaml

# Install Jaeger Operator
helm install jaeger-operator jaegertracing/jaeger-operator \
  --namespace observability --create-namespace

# Create Jaeger instance
kubectl apply -f k8s/monitoring/jaeger-instance.yaml

# Verify monitoring deployments
kubectl get pods -n monitoring
kubectl get pods -n observability
```

### 4. Access Services

After deployment, find the LoadBalancer IP addresses:

```bash
# Get API and UI endpoints
kubectl get svc -n asr-system

# Get Grafana endpoint
kubectl get svc -n monitoring prometheus-grafana

# Get Jaeger endpoint
kubectl get svc -n observability jaeger-query
```

**Access URLs:**
- ASR API: http://<api-loadbalancer-ip>
- ASR UI: http://<ui-loadbalancer-ip>
- Grafana: http://<grafana-loadbalancer-ip> (Username: admin, Password: admin)
- Jaeger UI: http://<jaeger-query-loadbalancer-ip>:16686

## CI/CD with Jenkins

The included Jenkinsfile automates:
1. Building Docker images
2. Pushing images to Docker Hub
3. Deploying to Kubernetes
4. Setting up monitoring

### Jenkins Requirements

- Jenkins with following plugins:
  - Docker Pipeline
  - Kubernetes
  - Credentials Binding
- Credentials:
  - `docker-hub-credentials`: Docker Hub username/password
  - `do-api-token`: Digital Ocean API token

## Clean Up

To destroy the infrastructure when no longer needed:

```bash
# Navigate to terraform directory
cd terraform

# Destroy resources
terraform destroy
```
