# Vietnamese ASR API Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the Vietnamese Automatic Speech Recognition (ASR) API.

## Components

- **api-configmap.yaml**: Configuration values for the ASR API
- **api-deployment.yaml**: Deployment specification for the ASR API
- **api-service.yaml**: Service definition for internal access
- **api-ingress.yaml**: Ingress definition for external access
- **api-hpa.yaml**: HorizontalPodAutoscaler for automatic scaling

## Prerequisites

- Kubernetes cluster (1.19+)
- kubectl CLI tool
- Docker image registry (if using custom images)
- NVIDIA GPU operator (for GPU support)
- Ingress controller (e.g., NGINX)

## Deployment Instructions

1. **Build and push the Docker image**:
   ```bash
   docker build -t your-registry/vietnamese-asr:latest .
   docker push your-registry/vietnamese-asr:latest
   ```

2. **Update image reference in api-deployment.yaml** if necessary.

3. **Deploy the resources**:
   ```bash
   # Create namespace
   kubectl create namespace asr

   # Apply ConfigMap
   kubectl apply -f api-configmap.yaml -n asr

   # Deploy the API
   kubectl apply -f api-deployment.yaml -n asr

   # Create the Service
   kubectl apply -f api-service.yaml -n asr

   # Create the HPA
   kubectl apply -f api-hpa.yaml -n asr

   # Deploy the Ingress
   kubectl apply -f api-ingress.yaml -n asr
   ```

4. **Verify the deployment**:
   ```bash
   kubectl get pods -n asr
   kubectl get services -n asr
   kubectl get ingress -n asr
   kubectl get hpa -n asr
   ```

## Monitoring

The ASR API is configured with Prometheus metrics endpoints. The deployment includes:

- Prometheus annotations for automatic service discovery
- Dashboard for Grafana to visualize metrics
- Alert rules for critical conditions

## Scaling

The HorizontalPodAutoscaler will automatically scale the deployment based on:
- CPU utilization (target: 75%)
- Memory utilization (target: 80%)

## Troubleshooting

Check pod logs:
```bash
kubectl logs -f -l app=asr-api -n asr
```

Check pod status:
```bash
kubectl describe pod -l app=asr-api -n asr
```

## Security

- Ensure proper network policies are in place
- Consider using Kubernetes secrets for sensitive configuration 