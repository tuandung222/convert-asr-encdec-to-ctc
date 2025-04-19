# Vietnamese ASR UI Application

A Streamlit-based user interface for Vietnamese Automatic Speech Recognition (ASR) that interacts with a FastAPI backend service.

## Features

- Upload audio files for transcription
- Record audio directly in the browser
- Select from multiple ASR models
- Language selection (Vietnamese, English, or auto-detect)
- Transcription history with analytics
- Confidence score visualization

## Requirements

- Python 3.8+
- Streamlit
- Dependencies listed in `requirements.txt`

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r ui/requirements.txt
   ```
3. Make sure the FastAPI backend is running

## Environment Variables

Create a `.env` file in the `ui` directory with the following variables:

```
API_URL=http://localhost:8000
```

## Running the Application

```bash
cd ui
streamlit run app.py
```

By default, the application will be available at http://localhost:8501

## Docker Support

Build and run using Docker:

```bash
docker build -t vietnamese-asr-ui -f ui/Dockerfile .
docker run -p 8501:8501 vietnamese-asr-ui
```

## API Integration

The UI connects to the Vietnamese ASR API which should be running and accessible. By default, it looks for the API at http://localhost:8000, but you can configure this using the API_URL environment variable.

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- OGG (.ogg)
- FLAC (.flac)
