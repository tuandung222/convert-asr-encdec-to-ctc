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

## Setting Up Jenkins for MLOps Docker Image Automation

This section provides a step-by-step guide for setting up Jenkins specifically to automate Docker image building and pushing to Docker Registry for small MLOps projects.

### 1. Set Up Jenkins Server

#### Quick Setup with Docker (Recommended for MLOps Projects)

```bash
# Create a volume for Jenkins data
docker volume create jenkins_data

# Run Jenkins with Docker capabilities
docker run -d --name jenkins-server \
  -p 8080:8080 -p 50000:50000 \
  -v jenkins_data:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --restart unless-stopped \
  jenkins/jenkins:lts
```

#### Install Required Plugins

After Jenkins is running:

1. Access Jenkins at http://localhost:8080
2. Get the initial admin password:
   ```bash
   docker exec jenkins-server cat /var/jenkins_home/secrets/initialAdminPassword
   ```
3. Follow the setup wizard and install suggested plugins
4. Install additional plugins from Manage Jenkins > Manage Plugins > Available:
   - Docker Pipeline
   - Docker Build Step
   - Git Integration
   - Pipeline Utility Steps
   - Blue Ocean (optional but recommended for better UI)

### 2. Configure Docker Credentials

1. Go to Jenkins > Manage Jenkins > Manage Credentials
2. Click on Jenkins store (global domain)
3. Click "Add Credentials"
4. Select "Username with password" for the kind
5. Enter your Docker Hub username and password/token
6. Set ID as "docker-hub-credentials" (must match Jenkinsfile)
7. Set Description as "Docker Hub Credentials"
8. Click OK

### 3. Set Up the Pipeline for Docker Image Builds

#### Option 1: Simple Docker Image Build Pipeline

Create a new Pipeline job:

1. Click "New Item" on the Jenkins dashboard
2. Enter a name like "vietnamese-asr-docker-build"
3. Select "Pipeline" and click OK
4. In the Pipeline section, select "Pipeline script from SCM"
5. Select Git as SCM
6. Enter your repository URL
7. Specify the branch (e.g., */main)
8. Set Script Path to "jenkins/Jenkinsfile"
9. Save the configuration

#### Option 2: Multibranch Pipeline

For projects with multiple branches:

1. Click "New Item" on the Jenkins dashboard
2. Enter a name like "vietnamese-asr-multibranch"
3. Select "Multibranch Pipeline" and click OK
4. In the Branch Sources section, click "Add source" and select "Git"
5. Enter your repository URL and credentials (if needed)
6. In the "Discover branches" section, select "All branches"
7. Set the Build Configuration to "by Jenkinsfile", with Script Path "jenkins/Jenkinsfile"
8. Save the configuration

### 4. Customize Build Configuration

For MLOps projects, you may want to simplify the Jenkinsfile. Edit the provided Jenkinsfile to focus on Docker image building and pushing:

```groovy
pipeline {
    agent any
    
    environment {
        DOCKER_HUB_CREDS = credentials('docker-hub-credentials')
        DOCKER_IMAGE_API = "tuandung12092002/asr-api"
        DOCKER_IMAGE_UI = "tuandung12092002/asr-ui"
        DOCKER_TAG = "${env.BUILD_NUMBER}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Docker Images') {
            steps {
                sh 'docker build -t ${DOCKER_IMAGE_API}:${DOCKER_TAG} -f Dockerfile .'
                sh 'docker build -t ${DOCKER_IMAGE_UI}:${DOCKER_TAG} -f ui/Dockerfile ui/'
                
                sh 'docker tag ${DOCKER_IMAGE_API}:${DOCKER_TAG} ${DOCKER_IMAGE_API}:latest'
                sh 'docker tag ${DOCKER_IMAGE_UI}:${DOCKER_TAG} ${DOCKER_IMAGE_UI}:latest'
            }
        }
        
        stage('Push Docker Images') {
            steps {
                sh 'echo $DOCKER_HUB_CREDS_PSW | docker login -u $DOCKER_HUB_CREDS_USR --password-stdin'
                
                sh 'docker push ${DOCKER_IMAGE_API}:${DOCKER_TAG}'
                sh 'docker push ${DOCKER_IMAGE_API}:latest'
                
                sh 'docker push ${DOCKER_IMAGE_UI}:${DOCKER_TAG}'
                sh 'docker push ${DOCKER_IMAGE_UI}:latest'
            }
        }
    }
    
    post {
        always {
            sh 'docker logout'
            sh 'docker system prune -f'
        }
    }
}
```

### 5. Set Up Webhook for Automatic Builds (Optional)

To automatically trigger builds when code is pushed to GitHub:

1. In Jenkins, install the "GitHub Integration" plugin
2. Configure your project to enable build triggers from GitHub webhook
3. In your GitHub repository, go to Settings > Webhooks
4. Add a webhook with the Payload URL: `http://<your-jenkins-url>/github-webhook/`
5. Set content type to `application/json`
6. Select "Just the push event"
7. Save the webhook

### 6. Common MLOps Workflow

For a typical MLOps workflow:

1. Develop and test your code locally
2. Push changes to GitHub
3. Jenkins automatically builds Docker images
4. Images are pushed to Docker Hub
5. You can then pull and deploy these images to your ML serving environment

### 7. Troubleshooting for MLOps Projects

- **Docker Socket Permission Issue**: If Jenkins can't access Docker, run:
  ```bash
  docker exec -it jenkins-server bash
  apt-get update && apt-get install -y sudo
  usermod -aG docker jenkins
  # Then restart the Jenkins container
  ```

- **Image Build Failures**: Check that your Dockerfiles are in the right locations and properly formatted

- **Docker Push Authentication Failures**: Verify your Docker Hub credentials in Jenkins

## Setting Up Jenkins (General Instructions)

### Using Docker

```bash
# Build the Jenkins Docker image
docker build -t speech-jenkins -f jenkins/Dockerfile jenkins/

# Run Jenkins container
docker run -d -p 8084:8080 -p 50000:50000 \
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
