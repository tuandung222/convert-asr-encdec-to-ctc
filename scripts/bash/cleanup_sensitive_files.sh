#!/bin/bash
# This script removes sensitive files from git tracking without deleting them locally

echo "Removing sensitive files from git tracking..."

# Remove terraform state files
git rm --cached terraform/terraform.tfstate 2>/dev/null || echo "terraform.tfstate not tracked"
git rm --cached terraform/terraform.tfstate.backup 2>/dev/null || echo "terraform.tfstate.backup not tracked"

# Remove kubeconfig
git rm --cached terraform/kubeconfig.yaml 2>/dev/null || echo "kubeconfig.yaml not tracked"
git rm --cached kubeconfig.yaml 2>/dev/null || echo "Root kubeconfig.yaml not tracked"

# Remove tfvars files
git rm --cached terraform/terraform.tfvars 2>/dev/null || echo "terraform.tfvars not tracked"
git rm --cached terraform/*.tfvars 2>/dev/null || echo "No other tfvars tracked"

echo "Creating commit to remove sensitive files..."
git commit -m "Remove sensitive files from git tracking"

echo "Done. You can now try pushing again."
echo "Remember to regularly check for sensitive files before pushing with:"
echo "git status"
