#!/bin/bash

# Vietnamese ASR Kubernetes Deployment

This directory contains configurations for deploying the Vietnamese ASR system on Digital Ocean Kubernetes Service (DOKS).

## Prerequisites

- [Digital Ocean](https://www.digitalocean.com/) account with API token
- [Terraform](https://www.terraform.io/) (v1.0+)
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- [Helm](https://helm.sh/) (v3.0+)
- [doctl](https://docs.digitalocean.com/reference/doctl/) - Digital Ocean CLI

## Deployment Structure

The deployment is organized into sequential steps, each with its own script:

1. **Infrastructure Setup** (1_infrastructure_setup.sh)
   - Creates Kubernetes cluster on Digital Ocean using Terraform
   - Sets up node pools and networking

2. **Kubernetes Configuration** (2_configure_kubernetes.sh)
   - Configures kubectl to communicate with the cluster
   - Sets up necessary authentication

3. **Monitoring Setup** (3_setup_monitoring.sh)
   - Installs Prometheus, Grafana, and Jaeger
   - Sets up dashboards and service monitoring

4. **Application Deployment** (4_deploy_application.sh)
   - Deploys the API and UI components
   - Configures services and endpoints

5. **Cleanup** (5_cleanup.sh)
   - Removes all resources when they're no longer needed

## Quick Deployment

For a guided deployment process that runs all steps in sequence:

```bash
# Make the scripts executable
chmod +x *.sh

# Run the main setup script
./setup.sh
```

## Step-by-Step Deployment

If you prefer to run each step manually:

### 1. Infrastructure Provisioning with Terraform

```bash
# Create the Kubernetes cluster on Digital Ocean
./1_infrastructure_setup.sh
```

### 2. Configure Kubernetes

```bash
# Set up kubectl to connect to the cluster
./2_configure_kubernetes.sh
```

### 3. Set Up Monitoring

```bash
# Install and configure Prometheus, Grafana, and Jaeger
./3_setup_monitoring.sh
```

### 4. Deploy Application

```bash
# Deploy the ASR API and UI components
./4_deploy_application.sh
```

### 5. Clean Up When Done

```bash
# Remove all resources when no longer needed
./5_cleanup.sh
```

## Access Services

After deployment, you have two options to access services:

### Option 1: Direct Service Access (Default)

By default, the main API and UI services are deployed with LoadBalancer type:

```bash
# Get LoadBalancer IPs
kubectl get svc -n asr-system
```

**Access URLs:**
- ASR API: http://<api-loadbalancer-ip>
- ASR UI: http://<ui-loadbalancer-ip>

### Option 2: Unified Access with Ingress (Recommended)

For a more convenient access pattern using a single IP address, deploy the NGINX Ingress Controller:

```bash
# From the k8s directory
cd ingress
./setup-ingress.sh
```

This will:
1. Deploy an NGINX Ingress Controller
2. Convert all services to ClusterIP
3. Set up path-based routing

**Access URLs with Ingress:**
```
http://<ingress-ip>/api      # ASR API
http://<ingress-ip>/ui       # ASR UI
http://<ingress-ip>/prometheus  # Prometheus 
http://<ingress-ip>/grafana     # Grafana (admin/admin)
http://<ingress-ip>/jaeger      # Jaeger UI
```

The Ingress approach is **recommended** because:
- Uses only one LoadBalancer (stays within Digital Ocean free tier limits)
- Provides consistent URLs
- Enables unified TLS/SSL setup
- Improves security by exposing fewer endpoints

## Security Notes

The deployment scripts handle Digital Ocean API tokens and other sensitive information. For security:

- Use environment variables when possible instead of storing tokens in files
- Never commit files containing API tokens or credentials to version control
- The scripts automatically update .gitignore to exclude sensitive files

## Troubleshooting

If you encounter issues during deployment, check:

1. Digital Ocean API token permissions
2. Cluster status in the Digital Ocean dashboard
3. Pod status with `kubectl get pods -A`
4. Logs with `kubectl logs <pod-name> -n <namespace>`

## Monitoring

The deployment includes a monitoring stack with Prometheus, Grafana, and Jaeger for observability:

- **Prometheus**: Collects metrics from the ASR system
- **Grafana**: Visualizes metrics with pre-configured dashboards
- **Jaeger**: Provides distributed tracing for API requests

Unlike the previous implementation that used Helm charts, this version deploys the monitoring components directly as Kubernetes manifests for easier customization and management.

### Monitoring Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Prometheus │     │   Grafana   │     │    Jaeger   │
│  (Metrics)  │     │ (Dashboards)│     │  (Tracing)  │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────┬───────┴───────────┬───────┘
                   │                   │
         ┌─────────▼─────────┐ ┌───────▼───────┐
         │    ASR API        │ │     ASR UI    │
         │ (instrumented)    │ │               │
         └───────────────────┘ └───────────────┘
```

### Accessing Monitoring Services

All monitoring services are accessible through NodePort services:

```bash
# Get the NodeIP and service ports
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')

# Access URLs 
Prometheus: http://<node-ip>:<prometheus-nodeport>
Grafana: http://<node-ip>:<grafana-nodeport> (admin/admin)
Jaeger UI: http://<node-ip>:<jaeger-query-nodeport>
```

For more information about the monitoring stack, see the [Monitoring Stack](../KUBERNETES.md#monitoring-stack) section in the main KUBERNETES.md documentation.
