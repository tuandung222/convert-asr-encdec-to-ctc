# ASR System Monitoring

This directory contains the configuration for monitoring the Vietnamese ASR system using Prometheus and Grafana.

## Components

- **Prometheus**: Collects metrics from all services
- **Grafana**: Creates dashboards and visualizations for the metrics
- **Node Exporter**: Collects system-level metrics
- **cAdvisor**: Collects container-level metrics

## Metrics

The monitoring system tracks several metrics:

- HTTP request rate and response time
- API memory and CPU usage
- Total transcriptions processed
- System-level metrics (CPU, memory, disk, network)
- Container-level metrics

## Dashboard

A pre-configured Grafana dashboard is available at `http://localhost:3000` after starting the system.

Default login:
- Username: admin
- Password: admin

## Directory Structure

- `prometheus/`: Contains Prometheus configuration
- `grafana/`: Contains Grafana configuration
  - `provisioning/`: Auto-provisioning configurations
    - `datasources/`: Prometheus data source configuration
    - `dashboards/`: Dashboard provisioning configuration
      - `json/`: Dashboard JSON definitions

## Adding New Metrics

To add new metrics to the ASR system:

1. Instrument your code with Prometheus metrics (using libraries like `prometheus_client` for Python)
2. Update the Prometheus configuration if needed
3. Create or update dashboards in Grafana

## Alerting

The system can be configured to send alerts based on metric thresholds.
To configure alerts:

1. Define alert rules in Prometheus
2. Configure notification channels in Grafana
3. Set up alerting on specific panels or dashboards
