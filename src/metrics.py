import time
from prometheus_client import Counter, Histogram, Gauge, Summary

# Define metrics
REQUESTS = Counter(
    'http_requests_total', 
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0]
)

TRANSCRIPTIONS = Counter(
    'transcription_count_total',
    'Total number of audio transcriptions',
    ['model', 'language', 'status']
)

TRANSCRIPTION_DURATION = Histogram(
    'transcription_duration_seconds',
    'Transcription processing time in seconds',
    ['model', 'language'],
    buckets=[0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0, 120.0]
)

AUDIO_DURATION = Summary(
    'audio_duration_seconds_summary',
    'Summary of audio file durations in seconds',
    ['format']
)

MODEL_LOADING_TIME = Histogram(
    'model_loading_time_seconds',
    'Time taken to load ASR model in seconds',
    ['model', 'checkpoint'],
    buckets=[0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0]
)

INFERENCE_IN_PROGRESS = Gauge(
    'inference_in_progress',
    'Number of ASR inference operations currently in progress',
    ['model']
)

MODEL_LOAD_FAILURES = Counter(
    'model_load_failures_total',
    'Number of times model loading has failed',
    ['model', 'type']
)

# Utility class for measuring duration
class Timer:
    def __init__(self, metric, labels=None):
        self.metric = metric
        self.labels = labels or {}
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start
        self.metric.labels(**self.labels).observe(duration)
        
# Example usage:
# 
# @app.get("/endpoint")
# async def endpoint():
#     # Track request count
#     REQUESTS.labels(method="GET", endpoint="/endpoint", status=200).inc()
#     
#     # Track request duration
#     with Timer(REQUEST_DURATION, {"method": "GET", "endpoint": "/endpoint"}):
#         # Your endpoint logic here
#         result = process_request()
#     
#     return result
#
# def transcribe_audio(audio_file, model, language):
#     # Track transcription count
#     TRANSCRIPTIONS.labels(model=model, language=language, status="started").inc()
#     
#     # Track inference in progress
#     INFERENCE_IN_PROGRESS.labels(model=model).inc()
#     
#     try:
#         # Track transcription time
#         with Timer(TRANSCRIPTION_DURATION, {"model": model, "language": language}):
#             # Your transcription logic here
#             result = process_transcription()
#         
#         # Track success
#         TRANSCRIPTIONS.labels(model=model, language=language, status="success").inc()
#         return result
#     except Exception as e:
#         # Track failure
#         TRANSCRIPTIONS.labels(model=model, language=language, status="failure").inc()
#         raise
#     finally:
#         # Decrement in-progress count
#         INFERENCE_IN_PROGRESS.labels(model=model).dec() 