# Main Terraform configuration file for ASR Kubernetes infrastructure on DigitalOcean
# This file defines the core infrastructure components for our Automatic Speech Recognition (ASR) system.
# It provisions a managed Kubernetes cluster on DigitalOcean with auto-scaling capabilities.

# Terraform configuration block specifying required providers
terraform {
  required_providers {
    # Define the DigitalOcean provider with version constraint
    # This provider allows Terraform to create and manage DigitalOcean resources
    digitalocean = {
      source  = "digitalocean/digitalocean" # Source of the provider in the Terraform Registry
      version = "~> 2.0"                    # Version constraint, allowing any 2.x version
    }
  }
}

# Configure the DigitalOcean provider with authentication token
# This authenticates Terraform to make API calls to DigitalOcean on our behalf
provider "digitalocean" {
  token = var.do_token # Use the API token defined in variables.tf
}

# Create a Kubernetes cluster on DigitalOcean (DOKS - DigitalOcean Kubernetes Service)
# This is the primary resource that will host our ASR application workloads
resource "digitalocean_kubernetes_cluster" "asr_cluster" {
  name         = "asr-k8s-cluster"                                 # Name of the Kubernetes cluster
  region       = var.region                                        # Region where the cluster will be deployed (default: sgp1)
  version      = var.kubernetes_version                            # Kubernetes version to use (from variables)
  auto_upgrade = true                                              # Enable automatic Kubernetes version upgrades
  tags         = ["asr-system", "production", "terraform-managed"] # Tags for organization and filtering

  # Define maintenance window for automatic upgrades
  # This specifies when DigitalOcean can perform maintenance operations
  # Scheduled during low-traffic periods to minimize disruption
  maintenance_policy {
    start_time = "04:00"  # Start maintenance at 4 AM
    day        = "sunday" # Perform maintenance on Sundays
  }

  # Define the worker node pool configuration
  # Worker nodes are the VMs that will run our containerized ASR workloads
  node_pool {
    name       = "worker-pool"      # Name of the node pool
    size       = var.node_size      # Size/type of the nodes (default: s-2vcpu-4gb)
    node_count = var.node_count     # Initial number of nodes (default: 3)

    auto_scale = true               # Enable auto-scaling for the node pool
    min_nodes  = 1                  # Minimum number of nodes when auto-scaling (required to be at least 1)
    # max_nodes  = var.node_count + 1 # Maximum nodes is one more than the initial count

    max_nodes = var.node_count # Maximum nodes is the initial count

    tags = ["asr-worker", "production"] # Tags specific to these worker nodes

    # Optional: Add taints if you need to reserve nodes for specific workloads
    # Taints prevent pods from being scheduled on nodes unless they have matching tolerations
    # Useful for dedicating nodes to specific workloads like GPU-intensive ASR processing
    # taint {
    #   key    = "workloadKind"
    #   value  = "asr"
    #   effect = "NoSchedule"
    # }
  }

  # Set a longer timeout for cluster creation
  # DigitalOcean K8s clusters can take time to provision, especially with multiple nodes
  timeouts {
    create = "30m" # Allow up to 30 minutes for cluster creation
  }
}

# Create a local kubeconfig file for connecting to the cluster
# This file will be used by kubectl to authenticate and connect to the cluster
# It enables DevOps teams to manage the cluster after Terraform provisioning
resource "local_file" "kubeconfig" {
  depends_on = [digitalocean_kubernetes_cluster.asr_cluster]                         # Ensure cluster exists before creating file
  content    = digitalocean_kubernetes_cluster.asr_cluster.kube_config[0].raw_config # Get kubeconfig from cluster
  filename   = "${path.module}/kubeconfig.yaml"                                      # Save to kubeconfig.yaml in the current module directory
}
