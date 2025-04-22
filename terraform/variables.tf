# This file defines the input variables for our Terraform configuration.
# Variables allow us to parameterize our infrastructure deployment and make it reusable.

# Digital Ocean API token required for authentication
# This token must have write access to create and manage resources
# The sensitive flag ensures this value won't be displayed in logs or console output
variable "do_token" {
  description = "Digital Ocean API token"
  type        = string
  sensitive   = true  # Marks this variable as sensitive to prevent accidental exposure
}

# The geographic region where the Kubernetes cluster will be deployed
# Default is set to Singapore (sgp1) but can be overridden in terraform.tfvars
# Other common options include: nyc1 (New York), fra1 (Frankfurt), lon1 (London)
variable "region" {
  description = "Digital Ocean region for the Kubernetes cluster"
  type        = string
  default     = "sgp1"  # Singapore region
}

# The version of Kubernetes to use for the Digital Ocean Kubernetes Service (DOKS) cluster
# This should be updated periodically to maintain security and feature updates
# Check Digital Ocean documentation for supported versions
variable "kubernetes_version" {
  description = "Kubernetes version for the DOKS cluster"
  type        = string
  default     = "1.32.2-do.0"  # Format: [kubernetes version]-do.[revision]
}

# The machine type/size for worker nodes in the Kubernetes cluster
# Format is s-[vCPU count]vcpu-[RAM in GB]gb
# Larger sizes are available for more resource-intensive workloads
variable "node_size" {
  description = "Size of the worker nodes"
  type        = string
  default     = "s-2vcpu-4gb"  # 2 vCPU, 4GB RAM
}

# The initial number of worker nodes to provision in the cluster
# This also sets the minimum number of nodes when auto-scaling is enabled
# The maximum nodes will be calculated as node_count * 2 in main.tf
variable "node_count" {
  description = "Number of worker nodes in the cluster"
  type        = number
  default     = 2  # A minimum of 2 nodes is recommended for production workloads
}
