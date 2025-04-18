pipeline {
    agent {
        docker {
            image 'python:3.9'
        }
    }
    
    environment {
        DOCKER_HUB_CREDS = credentials('docker-hub-credentials')
        DOCKER_IMAGE_API = "yourdockerhub/asr-api"
        DOCKER_TAG = "${env.BUILD_NUMBER}"
        K8S_CONFIG = credentials('k8s-config')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install pytest pytest-cov flake8'
            }
        }
        
        stage('Code Quality') {
            steps {
                sh 'flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics'
            }
        }
        
        stage('Test') {
            steps {
                sh 'pytest --cov=src tests/'
            }
            post {
                always {
                    junit 'test-results/*.xml'
                    cobertura coberturaReportFile: 'coverage.xml'
                }
            }
        }
        
        stage('Build Docker Images') {
            steps {
                sh 'docker build -t ${DOCKER_IMAGE_API}:${DOCKER_TAG} -f api/Dockerfile .'
                sh 'docker tag ${DOCKER_IMAGE_API}:${DOCKER_TAG} ${DOCKER_IMAGE_API}:latest'
            }
        }
        
        stage('Push Docker Images') {
            steps {
                sh 'echo $DOCKER_HUB_CREDS_PSW | docker login -u $DOCKER_HUB_CREDS_USR --password-stdin'
                sh 'docker push ${DOCKER_IMAGE_API}:${DOCKER_TAG}'
                sh 'docker push ${DOCKER_IMAGE_API}:latest'
            }
        }
        
        stage('Deploy to Kubernetes') {
            when {
                branch 'main'
            }
            steps {
                sh 'mkdir -p ~/.kube'
                sh 'echo "$K8S_CONFIG" > ~/.kube/config'
                sh 'sed -i "s|{{IMAGE_TAG}}|${DOCKER_TAG}|g" k8s/api-deployment.yaml'
                sh 'kubectl apply -f k8s/'
            }
        }
    }
    
    post {
        always {
            sh 'docker logout'
            cleanWs()
        }
        success {
            echo 'Build successful!'
        }
        failure {
            echo 'Build failed!'
        }
    }
} 