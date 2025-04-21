variable "do_token" {
  description = "Digital Ocean API token"
  type        = string
  sensitive   = true
}

variable "region" {
  description = "Digital Ocean region for the Kubernetes cluster"
  type        = string
  default     = "sgp1"  # Singapore region
}

variable "kubernetes_version" {
  description = "Kubernetes version for the DOKS cluster"
  type        = string
  default     = "1.26.3-do.0"
}

variable "node_size" {
  description = "Size of the worker nodes"
  type        = string
  default     = "s-2vcpu-4gb"  # 2 vCPU, 4GB RAM
}

variable "node_count" {
  description = "Number of worker nodes in the cluster"
  type        = number
  default     = 3
}
