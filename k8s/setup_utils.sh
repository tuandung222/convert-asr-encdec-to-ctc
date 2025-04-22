#!/bin/bash
# Common utilities for Vietnamese ASR Kubernetes setup scripts

# Define color codes for better readability in terminal output
GREEN='\033[0;32m'   # Success messages
YELLOW='\033[1;33m'  # Section headers and warnings
RED='\033[0;31m'     # Error messages
NC='\033[0m'         # No Color (resets formatting)

# Check if required tools are installed
check_prerequisites() {
    local missing_tools=0

    echo -e "${YELLOW}=== Checking prerequisites ===${NC}"

    if ! command -v doctl &> /dev/null; then
        echo -e "${RED}Error: doctl is not installed. Please install it first.${NC}"
        echo "See: https://docs.digitalocean.com/reference/doctl/how-to/install/"
        missing_tools=1
    fi

    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}Error: kubectl is not installed. Please install it first.${NC}"
        echo "See: https://kubernetes.io/docs/tasks/tools/install-kubectl/"
        missing_tools=1
    fi

    if ! command -v terraform &> /dev/null; then
        echo -e "${RED}Error: terraform is not installed. Please install it first.${NC}"
        echo "See: https://learn.hashicorp.com/tutorials/terraform/install-cli"
        missing_tools=1
    fi

    if ! command -v helm &> /dev/null; then
        echo -e "${RED}Error: helm is not installed. Please install it first.${NC}"
        echo "See: https://helm.sh/docs/intro/install/"
        missing_tools=1
    fi

    if [ $missing_tools -eq 1 ]; then
        return 1
    fi
    
    echo -e "${GREEN}All required tools are installed.${NC}"
    return 0
}

# Get DigitalOcean API token from environment or prompt
get_api_token() {
    if [ -z "$DO_API_TOKEN" ]; then
        # Try to load from parent directory .env file
        if [ -f "../.env" ]; then
            echo -e "${GREEN}Loading API token from .env file...${NC}"
            export $(grep -v '^#' ../.env | xargs)
        fi

        # Try to load from current directory .env file
        if [ -f "./.env" ]; then
            echo -e "${GREEN}Loading API token from .env file...${NC}"
            export $(grep -v '^#' ./.env | xargs)
        fi

        # If token is still not set, prompt the user
        if [ -z "$DO_API_TOKEN" ]; then
            read -p "Enter your Digital Ocean API token: " DO_API_TOKEN
            if [ -z "$DO_API_TOKEN" ]; then
                echo -e "${RED}Error: API token cannot be empty.${NC}"
                return 1
            fi
        fi
    fi
    
    return 0
}

# Display security warning about sensitive files
display_security_warning() {
    echo -e "${RED}=== SECURITY WARNING ===${NC}"
    echo -e "This process will generate files containing sensitive information:"
    echo -e "- terraform.tfstate (contains DigitalOcean tokens and credentials)"
    echo -e "- kubeconfig.yaml (contains cluster access credentials)"
    echo -e "- terraform.tfvars (contains your API token)"
    echo -e "\nThese files should NEVER be committed to version control."
    echo -e "Make sure these files are in your .gitignore before pushing to a repository.\n"
    
    # Verify .gitignore exists and contains the necessary patterns
    if [ ! -f "../.gitignore" ] || ! grep -q "terraform.tfstate" "../.gitignore"; then
        echo -e "${YELLOW}Creating/updating .gitignore to exclude sensitive files...${NC}"
        cat >> "../.gitignore" << EOF
# Terraform files
terraform/.terraform/
terraform/.terraform.lock.hcl
terraform/terraform.tfstate
terraform/terraform.tfstate.backup
terraform/terraform.tfvars
*.tfvars

# Kubernetes files
kubeconfig.yaml
**/kubeconfig
**/*.kubeconfig

# Environment files
.env
**/.env
EOF
        echo -e "${GREEN}Updated .gitignore${NC}"
    fi
}

# Make script files executable
make_scripts_executable() {
    local scripts=("$@")
    for script in "${scripts[@]}"; do
        chmod +x "$script"
        echo -e "${GREEN}Made $script executable${NC}"
    done
} 