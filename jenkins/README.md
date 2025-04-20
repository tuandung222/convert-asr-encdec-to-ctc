# Jenkins CI/CD for Vietnamese Speech Recognition

This directory contains all the necessary configurations and scripts to set up a Jenkins CI/CD pipeline for the Vietnamese Speech Recognition project. The pipeline automates building Docker images, running tests, and deploying to different environments.

## Directory Structure

```
jenkins/
├── Dockerfile                # Dockerfile to create Jenkins server with required plugins
├── Jenkinsfile               # Pipeline definition
├── jenkins-deployment.yml    # Kubernetes config to deploy Jenkins (if using K8s)
├── plugins.txt               # List of required Jenkins plugins
├── scripts/                  # Helper scripts
│   ├── build-image.sh        # Script to build Docker images
│   └── deploy.sh             # Script to deploy
└── README.md                 # Installation and usage instructions
```

## Setting Up Jenkins

### Using Docker

```bash
# Build the Jenkins Docker image
docker build -t speech-jenkins -f jenkins/Dockerfile jenkins/

# Run Jenkins container
docker run -d -p 8080:8080 -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --name speech-jenkins speech-jenkins
```

### Using Kubernetes

```bash
# Apply the Kubernetes deployment configuration
kubectl apply -f jenkins/jenkins-deployment.yml

# Access Jenkins through NodePort (port 30080) or set up an Ingress
```

## Jenkins Configuration

1. Access Jenkins at http://localhost:8080 (or server IP)
2. Follow the initial setup instructions
3. Configure Credentials:
   - Docker Registry credentials (ID: `docker-registry-credentials`)
   - GitHub credentials (if repository is private)
   - Kubernetes credentials (if deploying to Kubernetes)

## Creating the Pipeline

1. In Jenkins, create a new Pipeline job
2. Configure SCM to point to your repository
3. Set the Jenkinsfile path to `jenkins/Jenkinsfile`
4. Save and run the pipeline

## Pipeline Stages

The Jenkins pipeline includes the following stages:

1. **Checkout**: Retrieves the code from the repository
2. **Lint & Test**: Runs linting and testing
3. **Build Docker Image**: Builds Docker images for the application
4. **Push Docker Image**: Pushes the images to the Docker registry
5. **Deploy to Development**: Automatically deploys to the dev environment (for develop branch)
6. **Deploy to Production**: Deploys to production after manual confirmation (for main branch)

## Usage

- Each commit to the repository will automatically trigger the pipeline
- Monitor build progress in the Jenkins UI or Blue Ocean UI: http://jenkins-server/blue
- For manual deployments, use the "Build with Parameters" option

## Customization

Update the following files to customize your CI/CD pipeline:

- **Jenkinsfile**: Modify stages, environment variables, and deployment logic
- **plugins.txt**: Add or remove Jenkins plugins as needed
- **deploy.sh**: Customize deployment strategy
- **build-image.sh**: Modify Docker build process

## Troubleshooting

- Check Jenkins logs if the pipeline fails
- Ensure Docker is accessible from Jenkins
- Verify credentials are correctly configured
- For Kubernetes deployments, ensure the service account has sufficient permissions
