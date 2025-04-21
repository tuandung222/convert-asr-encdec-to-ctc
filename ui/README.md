# Vietnamese ASR - Streamlit UI

This directory contains the Streamlit user interface for the Vietnamese Automatic Speech Recognition system.

## Features

- Modern, responsive UI for interacting with the ASR service
- Upload audio files for transcription
- Record audio directly through the browser
- View transcription history and download results
- System status monitoring and integration with Grafana

## Structure

- `app.py` - Main application entry point
- `utils.py` - Utility functions for API communication and data processing
- `components.py` - UI components and layout functions
- `static/` - Static assets (CSS, images)

## Running the UI

### Prerequisites

- Python 3.10+
- API service running (default: http://localhost:8000)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Environment Variables

- `API_URL`: URL of the FastAPI server (default: http://localhost:8000)
- `GRAFANA_URL`: URL of the Grafana dashboard (default: http://localhost:3000)

## UI Improvements

The UI has been redesigned for improved user experience:

1. **Modular Code Structure**
   - Separated into utils, components, and main app
   - Better maintainability and organization

2. **Enhanced UI/UX**
   - Modern, clean design with custom CSS
   - Improved layout and visual hierarchy
   - Better feedback for user actions

3. **New Features**
   - Improved recording workflow
   - Better history management with export functionality
   - Enhanced system status display
   - Theme customization options

4. **Responsive Design**
   - Adapts to different screen sizes
   - Mobile-friendly interface

## Integration

The UI communicates with the API service using REST endpoints and displays results in a user-friendly format. It also integrates with Grafana for metrics visualization.

## Screenshots

(Screenshots will be added here)

## License

This project is licensed under the MIT License.
