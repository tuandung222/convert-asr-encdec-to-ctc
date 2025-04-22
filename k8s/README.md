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

After deployment, find the LoadBalancer IP addresses:

```bash
# Get all service endpoints
kubectl get svc -A
```

**Access URLs:**
- ASR API: http://<api-loadbalancer-ip>
- ASR UI: http://<ui-loadbalancer-ip>
- Grafana: http://<grafana-loadbalancer-ip> (Username: admin, Password: admin)
- Jaeger UI: http://<jaeger-query-loadbalancer-ip>:16686

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
