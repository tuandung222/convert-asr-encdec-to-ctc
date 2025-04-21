FROM jenkins/jenkins:lts

USER root

# Install dependencies
RUN apt-get update && \
    apt-get install -y docker.io python3 python3-pip && \
    apt-get clean

# Add jenkins user to docker group
RUN usermod -aG docker jenkins

# Switch to jenkins user
USER jenkins

# Install necessary plugins
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli -f /usr/share/jenkins/ref/plugins.txt

# Configure Jenkins Seed Job if needed
# Uncomment and create jobs directory when needed
# COPY jobs /usr/share/jenkins/ref/jobs/
