#!/bin/bash
# This script removes sensitive files from the entire git history
set -e  # Exit immediately if any command fails

echo "🔒 Removing sensitive files from git history..."
echo "⚠️ WARNING: This will rewrite git history!"
echo "Make sure any collaborators know they need to rebase their work after this."
read -p "Continue? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "Operation cancelled."
    exit 1
fi

# Create backup branch just in case
echo "Creating backup branch 'backup-before-clean'..."
git branch backup-before-clean

# Use filter-branch to remove sensitive files from the entire history
echo "Removing terraform state files from history..."
git filter-branch --force --index-filter \
    "git rm --cached --ignore-unmatch terraform/terraform.tfstate terraform/terraform.tfstate.backup terraform/kubeconfig.yaml terraform/terraform.tfvars" \
    --prune-empty --tag-name-filter cat -- --all

# Overwrite refs to ensure changes take effect
echo "Overwriting git references..."
git for-each-ref --format="delete %(refname)" refs/original/ | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "✅ History has been cleaned!"
echo ""
echo "Next steps:"
echo "1. Revoke your old DigitalOcean token (URGENT!)"
echo "2. Create a new token for future use"
echo "3. Force push to update remote repository:"
echo "   git push origin --force --all"
echo ""
echo "⚠️ IMPORTANT: Make sure your terraform.tfvars file is not tracked"
echo "   and your .gitignore is properly configured."
echo ""
echo "For future use, consider using environment variables instead:"
echo "export TF_VAR_do_token=\"your-new-token\""
echo "terraform apply"
