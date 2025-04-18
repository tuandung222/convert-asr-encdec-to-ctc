FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ffmpeg for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY ui/ .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_URL=http://fastapi:8000

# Run the Streamlit application
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py", "--server.address=0.0.0.0"] 