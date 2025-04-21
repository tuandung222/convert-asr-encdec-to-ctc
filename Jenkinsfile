pipeline {
    agent any

    environment {
        DOCKER_HUB_CREDS = credentials('docker-hub-credentials')
        DO_API_TOKEN = credentials('do-api-token')
        CLUSTER_NAME = 'asr-k8s-cluster'
        DOCKER_REGISTRY = 'tuandung12092002'
        API_IMAGE = 'asr-fastapi-server'
        UI_IMAGE = 'asr-streamlit-ui'
        TAG = sh(script: 'date +"%Y%m%d_%H%M%S"', returnStdout: true).trim()
        LATEST_TAG = 'latest'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Images') {
            steps {
                sh 'chmod +x ./push_images.sh'

                // Login to Docker Hub
                sh 'echo $DOCKER_HUB_CREDS_PSW | docker login -u $DOCKER_HUB_CREDS_USR --password-stdin'

                // Build and push images using the script
                sh './push_images.sh'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                // Authenticate with Digital Ocean
                sh 'doctl auth init -t $DO_API_TOKEN'

                // Get kubeconfig for the cluster
                sh 'doctl kubernetes cluster kubeconfig save $CLUSTER_NAME'

                // Apply Kubernetes manifests
                sh 'kubectl apply -f k8s/monitoring/observability-namespace.yaml'
                sh 'kubectl apply -f k8s/base/namespace.yaml'
                sh 'kubectl apply -f k8s/base/'

                // Restart deployments to pick up new images
                sh 'kubectl rollout restart deployment/asr-api -n asr-system'
                sh 'kubectl rollout restart deployment/asr-ui -n asr-system'

                // Wait for rollout to complete
                sh 'kubectl rollout status deployment/asr-api -n asr-system'
                sh 'kubectl rollout status deployment/asr-ui -n asr-system'
            }
        }

        stage('Deploy Monitoring') {
            steps {
                // Install Prometheus Stack
                sh '''
                    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
                    helm repo update
                    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
                        --namespace monitoring --create-namespace \
                        --values k8s/monitoring/prometheus-values.yaml
                '''

                // Install Jaeger Operator
                sh '''
                    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
                    helm repo update
                    helm upgrade --install jaeger-operator jaegertracing/jaeger-operator \
                        --namespace observability --create-namespace
                    kubectl apply -f k8s/monitoring/jaeger-instance.yaml
                '''
            }
        }

        stage('Verify Deployment') {
            steps {
                // Get service endpoints
                sh 'kubectl get services -n asr-system'
                sh 'kubectl get services -n monitoring'
                sh 'kubectl get services -n observability'

                // Check pods are running
                sh 'kubectl get pods -n asr-system'
                sh 'kubectl get pods -n monitoring'
                sh 'kubectl get pods -n observability'
            }
        }
    }

    post {
        always {
            sh 'docker logout'
        }
        success {
            echo 'Deployment completed successfully!'
        }
        failure {
            echo 'Deployment failed!'
        }
    }
}
