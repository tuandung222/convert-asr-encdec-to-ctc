pipeline {
    agent {
        docker {
            image 'python:3.10'
        }
    }

    environment {
        DOCKER_HUB_CREDS = credentials('docker-hub-credentials')
        REGISTRY = "tuandung12092002"
        DOCKER_IMAGE_API = "${REGISTRY}/asr-fastapi-server"
        DOCKER_IMAGE_UI = "${REGISTRY}/asr-streamlit-ui"
        DOCKER_TAG = "${env.BUILD_NUMBER}"
        DOCKER_LATEST_TAG = "latest"
        // K8S_CONFIG = credentials('k8s-config')
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

        // stage('Test') {
        //     steps {
        //         sh 'pytest --cov=src tests/'
        //     }
        //     post {
        //         always {
        //             junit 'test-results/*.xml'
        //             cobertura coberturaReportFile: 'coverage.xml'
        //         }
        //     }
        // }

        stage('Build Docker Images') {
            steps {
                // API image with proper name
                sh """
                docker build -t ${DOCKER_IMAGE_API}:${DOCKER_TAG} \
                  -t ${DOCKER_IMAGE_API}:${DOCKER_LATEST_TAG} \
                  --build-arg APP_USER=api \
                  --build-arg APP_USER_UID=1000 \
                  -f api/Dockerfile .
                """

                // UI image with proper name
                sh """
                docker build -t ${DOCKER_IMAGE_UI}:${DOCKER_TAG} \
                  -t ${DOCKER_IMAGE_UI}:${DOCKER_LATEST_TAG} \
                  --build-arg APP_USER=streamlit \
                  --build-arg APP_USER_UID=1000 \
                  -f ui/Dockerfile .
                """
            }
        }

        stage('Push Docker Images') {
            steps {
                // Login to Docker Hub
                sh 'echo $DOCKER_HUB_CREDS_PSW | docker login -u $DOCKER_HUB_CREDS_USR --password-stdin'

                // Push API images with both version tag and latest
                sh """
                docker push ${DOCKER_IMAGE_API}:${DOCKER_TAG}
                docker push ${DOCKER_IMAGE_API}:${DOCKER_LATEST_TAG}
                """

                // Push UI images with both version tag and latest
                sh """
                docker push ${DOCKER_IMAGE_UI}:${DOCKER_TAG}
                docker push ${DOCKER_IMAGE_UI}:${DOCKER_LATEST_TAG}
                """
                
                // Log build information
                echo "Successfully pushed images to registry:"
                echo "API: ${DOCKER_IMAGE_API}:${DOCKER_TAG}"
                echo "UI: ${DOCKER_IMAGE_UI}:${DOCKER_TAG}"
            }
        }

        // stage('Deploy to Development') {
        //     when {
        //         branch 'develop'
        //     }
        //     steps {
        //         sh 'mkdir -p ~/.kube'
        //         sh 'echo "$K8S_CONFIG" > ~/.kube/config'
        //         sh './jenkins/scripts/deploy.sh development'
        //     }
        // }

        // stage('Deploy to Production') {
        //     when {
        //         branch 'main'
        //     }
        //     steps {
        //         input message: 'Confirm deployment to Production environment?'
        //         sh 'mkdir -p ~/.kube'
        //         sh 'echo "$K8S_CONFIG" > ~/.kube/config'
        //         sh './jenkins/scripts/deploy.sh production'
        //     }
        // }
    }

    post {
        always {
            sh 'docker logout'
            sh 'docker image prune -f'
            cleanWs()
        }
        success {
            echo 'Build and deployment successful!'
            echo "API Image: ${DOCKER_IMAGE_API}:${DOCKER_TAG}"
            echo "UI Image: ${DOCKER_IMAGE_UI}:${DOCKER_TAG}"
            // You can add notification steps here (Slack, email, etc.)
        }
        failure {
            echo 'Build or deployment failed!'
            // You can add notification steps here (Slack, email, etc.)
        }
    }
}