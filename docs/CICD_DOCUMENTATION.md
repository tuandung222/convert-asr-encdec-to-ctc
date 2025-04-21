# CI/CD Documentation for Vietnamese Speech Recognition Project

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline implemented for the Vietnamese Speech Recognition project. The pipeline automates the process of building, testing, and deploying the application to different environments.

## CI/CD Architecture

The CI/CD pipeline is built using Jenkins and consists of the following components:

1. **Jenkins Server**: Orchestrates the CI/CD process
2. **Docker Registry**: Stores Docker images
3. **Kubernetes Cluster**: Deployment target for the application
4. **GitHub Repository**: Source code repository with webhook integration

## Jenkins Pipeline

The Jenkins pipeline is defined in the `Jenkinsfile` at the root of the project. The pipeline consists of the following stages:

1. **Checkout**: Retrieves the latest code from the Git repository
2. **Install Dependencies**: Installs Python dependencies required for testing
3. **Code Quality**: Runs code quality checks using Flake8 and pre-commit hooks
4. **Test**: Runs unit tests with pytest and generates code coverage reports
5. **Build Docker Images**: Builds Docker images for the API, UI, and Gradio components
6. **Push Docker Images**: Pushes the images to Docker Hub
7. **Deploy to Development**: Automatically deploys to the development environment when changes are pushed to the `develop` branch
8. **Deploy to Production**: Deploys to the production environment after manual approval when changes are pushed to the `main` branch

## Jenkins Configuration

The Jenkins server is configured with the following plugins and settings:

1. **Plugins**:
   - Docker Workflow
   - Blue Ocean
   - Pipeline Utility Steps
   - Git Integration
   - Kubernetes Plugin
   - Credentials Binding
   - Pipeline Stage View

2. **Credentials**:
   - Docker Hub credentials (`docker-hub-credentials`)
   - Kubernetes configuration (`k8s-config`)
   - GitHub credentials (if repository is private)

3. **Pipeline Job Configuration**:
   - SCM: Git
   - Repository URL: Your GitHub repository URL
   - Branch Specifier: `*/main`, `*/develop`
   - Script Path: `Jenkinsfile`

## Docker Images

The pipeline builds the following Docker images:

1. **ASR API Image (`tuandung12092002/asr-api`)**:
   - Contains the FastAPI server implementation
   - Exposes port 8000
   - Uses the ASR model for speech recognition

2. **UI Image (`tuandung12092002/asr-ui`)**:
   - Contains the Streamlit UI implementation
   - Exposes port 8501
   - Communicates with the API service

3. **Gradio Image (`tuandung12092002/asr-gradio`)**:
   - Contains the Gradio demo implementation
   - Exposes port 7860
   - Provides a simplified UI for testing

## Kubernetes Deployment

The application is deployed to a Kubernetes cluster with the following configurations:

1. **Namespaces**:
   - `speech-processing-dev`: Development environment
   - `speech-processing-prod`: Production environment

2. **Deployments**:
   - `speech-api`: API service with 1 replica in dev, 3 replicas in prod
   - `speech-ui`: UI service with 1 replica in dev, 2 replicas in prod

3. **Services**:
   - `speech-api-service`: Exposes the API on port 8000
   - `speech-ui-service`: Exposes the UI on port 8501

4. **Ingress**:
   - `speech-ingress`: Routes traffic based on path
     - `/api/*` → API service
     - `/*` → UI service

5. **Resource Allocation**:
   - **Development**:
     - API: 500m CPU, 512Mi memory (limits)
     - UI: 200m CPU, 256Mi memory (limits)
   - **Production**:
     - API: 1 CPU, 2Gi memory (limits)
     - UI: 500m CPU, 512Mi memory (limits)

## Deployment Process

The deployment process is managed by scripts in the `jenkins/scripts/` directory:

1. **`build-image.sh`**:
   - Builds Docker images for all components
   - Tags images with build number and "latest"

2. **`deploy.sh`**:
   - Deploys the application to Kubernetes
   - Takes environment (`development` or `production`) as parameter
   - Applies Kubernetes configurations based on environment
   - Updates image tags

## Monitoring

The production environment includes monitoring with Prometheus and Grafana:

1. **Prometheus**:
   - Scrapes metrics from the API service
   - Collects performance metrics
   - Tracks API request counts, latency, and errors

2. **Grafana**:
   - Provides dashboards for visualizing metrics
   - Includes pre-configured dashboards for the ASR system
   - Allows for monitoring real-time performance

## Security Considerations

The CI/CD pipeline includes the following security measures:

1. **Credentials Management**:
   - All credentials are stored securely in Jenkins
   - Credentials are injected into the pipeline at runtime
   - No credentials are stored in the repository

2. **Image Security**:
   - Docker images are built from trusted base images
   - Images are scanned for vulnerabilities (future enhancement)
   - Images are tagged with build numbers for traceability

3. **Deployment Security**:
   - Production deployments require manual approval
   - Kubernetes configurations use appropriate security contexts
   - TLS is enabled for production ingress

## Continuous Improvement

The CI/CD pipeline is designed for continuous improvement:

1. **Future Enhancements**:
   - Add automated vulnerability scanning
   - Implement canary deployments
   - Add integration and end-to-end tests
   - Set up automated rollbacks on failure

2. **Performance Optimization**:
   - Optimize Docker build process with caching
   - Implement parallel testing
   - Optimize Kubernetes resource allocation

## Troubleshooting

Common issues and their solutions:

1. **Pipeline Failures**:
   - Check Jenkins logs for detailed error messages
   - Verify that all required credentials are configured
   - Ensure Docker is accessible from Jenkins

2. **Deployment Failures**:
   - Check Kubernetes logs with `kubectl logs`
   - Verify that Kubernetes configuration is valid
   - Check resource constraints and adjust if necessary

3. **Image Build Failures**:
   - Check Docker build logs
   - Verify that Dockerfile is valid
   - Ensure all required files are present in the build context

## Conclusion

The CI/CD pipeline for the Vietnamese Speech Recognition project automates the process of building, testing, and deploying the application. It ensures consistent quality and facilitates rapid delivery of new features and bug fixes.

For more information, refer to the documentation in the `jenkins/` directory and the Kubernetes configurations in the `k8s/` directory.
