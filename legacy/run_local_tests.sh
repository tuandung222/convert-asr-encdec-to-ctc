#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Default values
API_URL="http://localhost:8000"
UI_URL="http://localhost:8501"
EXAMPLES_DIR="examples"
RESULTS_DIR="test_results"
PERF_RESULTS_DIR="$RESULTS_DIR/performance"
RUN_API_TESTS=true
RUN_UI_TESTS=false  # Disabled by default as it requires Selenium and browser
RUN_PERF_TESTS=true

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --no-api           Skip API tests"
    echo "  --ui               Run UI tests (requires Selenium and Chrome)"
    echo "  --no-perf          Skip performance tests"
    echo "  --api-url URL      Set API URL (default: $API_URL)"
    echo "  --ui-url URL       Set UI URL (default: $UI_URL)"
    echo "  --examples DIR     Set examples directory (default: $EXAMPLES_DIR)"
    echo "  --results DIR      Set results directory (default: $RESULTS_DIR)"
    echo "  --help             Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-api)
            RUN_API_TESTS=false
            shift
            ;;
        --ui)
            RUN_UI_TESTS=true
            shift
            ;;
        --no-perf)
            RUN_PERF_TESTS=false
            shift
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --ui-url)
            UI_URL="$2"
            shift 2
            ;;
        --examples)
            EXAMPLES_DIR="$2"
            shift 2
            ;;
        --results)
            RESULTS_DIR="$2"
            PERF_RESULTS_DIR="$RESULTS_DIR/performance"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Create results directory
mkdir -p "$RESULTS_DIR"
mkdir -p "$PERF_RESULTS_DIR"

echo -e "${GREEN}Starting Vietnamese ASR testing suite...${NC}"
echo "API URL: $API_URL"
echo "UI URL: $UI_URL"
echo "Examples directory: $EXAMPLES_DIR"
echo "Results directory: $RESULTS_DIR"
echo ""

# Check if examples directory exists
if [[ ! -d "$EXAMPLES_DIR" ]]; then
    echo -e "${RED}Error: Examples directory '$EXAMPLES_DIR' does not exist${NC}"
    echo "Creating examples directory..."
    mkdir -p "$EXAMPLES_DIR"
    echo -e "${YELLOW}Warning: No audio examples found. Some tests may fail.${NC}"
    echo "You should add .wav or .mp3 files to the examples directory."
fi

# Check for at least one audio file
AUDIO_FILES=(`find "$EXAMPLES_DIR" -type f \( -name "*.wav" -o -name "*.mp3" \)`)
if [[ ${#AUDIO_FILES[@]} -eq 0 ]]; then
    echo -e "${YELLOW}Warning: No audio files found in '$EXAMPLES_DIR'${NC}"
    echo "You should add .wav or .mp3 files to the examples directory."
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 is required but not installed${NC}"
    exit 1
fi

# Install required packages
echo -e "${GREEN}Installing required packages...${NC}"
pip install requests matplotlib numpy selenium &> /dev/null

# Run API tests
if [[ "$RUN_API_TESTS" = true ]]; then
    echo -e "\n${GREEN}Running API tests...${NC}"
    python3 test_api_server.py --url "$API_URL" --audio-dir "$EXAMPLES_DIR" --output "$RESULTS_DIR/api_test_results.json"

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}API tests completed successfully${NC}"
    else
        echo -e "${YELLOW}API tests completed with errors${NC}"
    fi

    echo "Results saved to $RESULTS_DIR/api_test_results.json"
fi

# Run UI tests
if [[ "$RUN_UI_TESTS" = true ]]; then
    echo -e "\n${GREEN}Running UI tests...${NC}"
    echo -e "${YELLOW}Note: UI tests require Selenium and Chrome/Chromium${NC}"

    python3 test_streamlit_ui.py --url "$UI_URL" --audio-dir "$EXAMPLES_DIR" --output "$RESULTS_DIR/ui_test_results.json"

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}UI tests completed successfully${NC}"
    else
        echo -e "${YELLOW}UI tests completed with errors${NC}"
    fi

    echo "Results saved to $RESULTS_DIR/ui_test_results.json"
fi

# Run performance tests
if [[ "$RUN_PERF_TESTS" = true ]]; then
    echo -e "\n${GREEN}Running performance tests...${NC}"

    # Get the first audio file for performance testing
    if [[ ${#AUDIO_FILES[@]} -gt 0 ]]; then
        AUDIO_FILE="${AUDIO_FILES[0]}"
        echo "Using audio file: $AUDIO_FILE"

        python3 test_performance.py --url "$API_URL" --audio-file "$AUDIO_FILE" --output "$RESULTS_DIR/performance_results.json" --graphs-dir "$PERF_RESULTS_DIR"

        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}Performance tests completed successfully${NC}"
        else
            echo -e "${YELLOW}Performance tests completed with errors${NC}"
        fi

        echo "Results saved to $RESULTS_DIR/performance_results.json"
        echo "Graphs saved to $PERF_RESULTS_DIR/"
    else
        echo -e "${RED}Error: No audio files available for performance testing${NC}"
    fi
fi

echo -e "\n${GREEN}All tests completed${NC}"
echo "Test results are available in the $RESULTS_DIR directory"

# Make the test scripts executable
chmod +x test_api_server.py test_streamlit_ui.py test_performance.py
