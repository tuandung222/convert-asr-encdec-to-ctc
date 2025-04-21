terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

resource "digitalocean_kubernetes_cluster" "asr_cluster" {
  name    = "asr-k8s-cluster"
  region  = var.region
  version = var.kubernetes_version

  node_pool {
    name       = "worker-pool"
    size       = var.node_size
    node_count = var.node_count
  }
}

# Create a kubeconfig file for connecting to the cluster
resource "local_file" "kubeconfig" {
  depends_on = [digitalocean_kubernetes_cluster.asr_cluster]
  content    = digitalocean_kubernetes_cluster.asr_cluster.kube_config[0].raw_config
  filename   = "${path.module}/kubeconfig.yaml"
}
