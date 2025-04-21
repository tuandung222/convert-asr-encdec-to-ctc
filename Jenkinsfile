pipeline {
    agent {
        docker {
            image 'python:3.10'
        }
    }

    environment {
        DOCKER_HUB_CREDS = credentials('docker-hub-credentials')
        DOCKER_IMAGE_API = "tuandung12092002/asr-api"
        DOCKER_IMAGE_UI = "tuandung12092002/asr-ui"
        // DOCKER_IMAGE_GRADIO = "tuandung12092002/asr-gradio"
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
                sh 'pre-commit run --all-files || true'  // Run pre-commit but don't fail if some checks fail
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
                // Main API image
                sh 'docker build -t ${DOCKER_IMAGE_API}:${DOCKER_TAG} -f Dockerfile .'
                sh 'docker tag ${DOCKER_IMAGE_API}:${DOCKER_TAG} ${DOCKER_IMAGE_API}:latest'

                // UI image
                sh 'docker build -t ${DOCKER_IMAGE_UI}:${DOCKER_TAG} -f ui/Dockerfile ui/'
                sh 'docker tag ${DOCKER_IMAGE_UI}:${DOCKER_TAG} ${DOCKER_IMAGE_UI}:latest'

                // Gradio image
                // sh 'docker build -t ${DOCKER_IMAGE_GRADIO}:${DOCKER_TAG} -f src/app/Dockerfile.gradio .'
                // sh 'docker tag ${DOCKER_IMAGE_GRADIO}:${DOCKER_TAG} ${DOCKER_IMAGE_GRADIO}:latest'
            }
        }

        stage('Push Docker Images') {
            steps {
                sh 'echo $DOCKER_HUB_CREDS_PSW | docker login -u $DOCKER_HUB_CREDS_USR --password-stdin'

                // Push API images
                sh 'docker push ${DOCKER_IMAGE_API}:${DOCKER_TAG}'
                sh 'docker push ${DOCKER_IMAGE_API}:latest'

                // Push UI images
                sh 'docker push ${DOCKER_IMAGE_UI}:${DOCKER_TAG}'
                sh 'docker push ${DOCKER_IMAGE_UI}:latest'

                // Push Gradio images
                // sh 'docker push ${DOCKER_IMAGE_GRADIO}:${DOCKER_TAG}'
                // sh 'docker push ${DOCKER_IMAGE_GRADIO}:latest'
            }
        }

        stage('Deploy to Development') {
            when {
                branch 'develop'
            }
            steps {
                sh 'mkdir -p ~/.kube'
                sh 'echo "$K8S_CONFIG" > ~/.kube/config'
                sh './jenkins/scripts/deploy.sh development'
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Confirm deployment to Production environment?'
                sh 'mkdir -p ~/.kube'
                sh 'echo "$K8S_CONFIG" > ~/.kube/config'
                sh './jenkins/scripts/deploy.sh production'
            }
        }
    }

    post {
        always {
            sh 'docker logout'
            sh 'docker image prune -f'
            cleanWs()
        }
        success {
            echo 'Build and deployment successful!'
            // You can add notification steps here (Slack, email, etc.)
        }
        failure {
            echo 'Build or deployment failed!'
            // You can add notification steps here (Slack, email, etc.)
        }
    }
}
