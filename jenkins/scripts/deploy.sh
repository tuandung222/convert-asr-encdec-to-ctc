#!/bin/bash

ENVIRONMENT=$1
NAMESPACE="speech-processing"

if [ "$ENVIRONMENT" == "production" ]; then
    NAMESPACE="speech-processing-prod"
elif [ "$ENVIRONMENT" == "development" ]; then
    NAMESPACE="speech-processing-dev"
else
    echo "Invalid environment. Use 'development' or 'production'"
    exit 1
fi

# Ensure namespace exists
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy application
kubectl -n $NAMESPACE apply -f k8s/base
kubectl -n $NAMESPACE apply -f k8s/$ENVIRONMENT

# Update image
kubectl -n $NAMESPACE set image deployment/speech-api speech-api=${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

echo "Successfully deployed to $ENVIRONMENT environment (namespace: $NAMESPACE)"
