# This file defines the outputs from our Terraform configuration.
# Outputs are values that are made available after Terraform applies the configuration.
# They can be used by other Terraform configurations, scripts, or displayed to users.

# The unique identifier for the DigitalOcean Kubernetes cluster
# This ID can be used with the DigitalOcean API or CLI tools
output "cluster_id" {
  description = "ID of the created Kubernetes cluster"
  value       = digitalocean_kubernetes_cluster.asr_cluster.id
}

# The URL endpoint for the Kubernetes API server
# This is used by kubectl and other tools to communicate with the cluster
output "cluster_endpoint" {
  description = "Endpoint for the Kubernetes API server"
  value       = digitalocean_kubernetes_cluster.asr_cluster.endpoint
}

# The local filesystem path to the generated kubeconfig file
# This file contains authentication details needed to connect to the cluster
output "kubeconfig_path" {
  description = "Path to the kubeconfig file"
  value       = "${path.module}/kubeconfig.yaml"
}

# The current status of the Kubernetes cluster (e.g., "running")
# Useful for verifying the cluster is operational
output "cluster_status" {
  description = "Status of the cluster"
  value       = digitalocean_kubernetes_cluster.asr_cluster.status
}

# The ID of the default node pool in the cluster
# A node pool is a group of worker nodes with the same configuration
output "node_pool_id" {
  description = "ID of the default node pool"
  value       = digitalocean_kubernetes_cluster.asr_cluster.node_pool[0].id
}

# The URN (Uniform Resource Name) of the cluster
# This is a unique identifier used by DigitalOcean for resource management
output "cluster_urn" {
  description = "Uniform Resource Name (URN) of the cluster"
  value       = digitalocean_kubernetes_cluster.asr_cluster.urn
}

# A ready-to-use command that configures kubectl to use this cluster
# Users can copy and paste this command to set up their local environment
output "kubectl_connect_command" {
  description = "Command to configure kubectl to connect to the cluster"
  value       = "export KUBECONFIG=${path.module}/kubeconfig.yaml"
}
