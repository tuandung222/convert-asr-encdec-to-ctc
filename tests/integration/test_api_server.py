#!/usr/bin/env python

"""
Local test script for the Vietnamese ASR FastAPI server
This script sends various test requests to the FastAPI server to verify its functionality
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ASRAPITester:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.results = {
            "health_check": None,
            "models_list": None,
            "languages_list": None,
            "transcriptions": [],
            "errors": [],
        }

    def test_health(self):
        """Test the health check endpoint"""
        logger.info("Testing health check endpoint...")
        try:
            response = requests.get(f"{self.api_url}/health")
            response.raise_for_status()
            self.results["health_check"] = response.json()
            logger.info(f"Health check: {response.json()}")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            self.results["errors"].append({"endpoint": "health", "error": str(e)})
            return False

    def test_models(self):
        """Test the models endpoint"""
        logger.info("Testing models endpoint...")
        try:
            response = requests.get(f"{self.api_url}/models")
            response.raise_for_status()
            self.results["models_list"] = response.json()
            logger.info(f"Models available: {response.json()}")
            return True
        except Exception as e:
            logger.error(f"Models endpoint failed: {str(e)}")
            self.results["errors"].append({"endpoint": "models", "error": str(e)})
            return False

    def test_languages(self):
        """Test the languages endpoint"""
        logger.info("Testing languages endpoint...")
        try:
            response = requests.get(f"{self.api_url}/languages")
            response.raise_for_status()
            self.results["languages_list"] = response.json()
            logger.info(f"Languages available: {response.json()}")
            return True
        except Exception as e:
            logger.error(f"Languages endpoint failed: {str(e)}")
            self.results["errors"].append({"endpoint": "languages", "error": str(e)})
            return False

    def test_transcribe(self, audio_file, model="phowhisper-tiny-ctc", language="vi"):
        """Test the transcribe endpoint with an audio file"""
        logger.info(f"Testing transcribe endpoint with file: {audio_file}")

        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            self.results["errors"].append(
                {"endpoint": "transcribe", "error": f"File not found: {audio_file}"}
            )
            return False

        try:
            with open(audio_file, "rb") as f:
                files = {"file": (os.path.basename(audio_file), f, "audio/wav")}
                data = {"model": model, "language": language}

                start_time = time.time()
                response = requests.post(f"{self.api_url}/transcribe", files=files, data=data)
                response.raise_for_status()
                end_time = time.time()

                result = response.json()
                result["test_response_time"] = end_time - start_time
                result["audio_file"] = audio_file

                self.results["transcriptions"].append(result)
                logger.info(f"Transcription result: {result['text']}")
                logger.info(f"Response time: {result['test_response_time']:.2f}s")
                logger.info(f"Real-time factor: {result['real_time_factor']:.2f}x")
                return True
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            self.results["errors"].append(
                {"endpoint": "transcribe", "error": str(e), "audio_file": audio_file}
            )
            return False

    def test_error_handling(self, invalid_file=None):
        """Test error handling with invalid requests"""
        logger.info("Testing error handling...")

        # Test with non-existent model
        try:
            with open(invalid_file or __file__, "rb") as f:
                files = {"file": (os.path.basename(f.name), f, "text/plain")}
                data = {"model": "non-existent-model", "language": "vi"}

                response = requests.post(f"{self.api_url}/transcribe", files=files, data=data)

                if response.status_code == 400:
                    logger.info("Non-existent model test passed")
                else:
                    logger.warning(f"Expected 400 status code, got {response.status_code}")
        except Exception as e:
            logger.error(f"Error handling test failed: {str(e)}")
            self.results["errors"].append({"endpoint": "error_handling", "error": str(e)})

        # Test with invalid file format
        try:
            with open(invalid_file or __file__, "rb") as f:
                files = {"file": (os.path.basename(f.name), f, "text/plain")}
                data = {"model": "phowhisper-tiny-ctc", "language": "vi"}

                response = requests.post(f"{self.api_url}/transcribe", files=files, data=data)

                if response.status_code >= 400:
                    logger.info("Invalid file format test passed")
                else:
                    logger.warning(f"Expected error status code, got {response.status_code}")
        except Exception as e:
            logger.error(f"Error handling test failed: {str(e)}")
            self.results["errors"].append({"endpoint": "error_handling", "error": str(e)})

    def run_all_tests(self, audio_files):
        """Run all tests in sequence"""
        logger.info("Running all tests...")

        if not self.test_health():
            logger.error("Health check failed, stopping tests")
            return False

        self.test_models()
        self.test_languages()

        for audio_file in audio_files:
            self.test_transcribe(audio_file)

        self.test_error_handling()

        return len(self.results["errors"]) == 0

    def save_results(self, output_file="api_test_results.json"):
        """Save test results to a file"""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Test results saved to {output_file}")


def find_audio_files(directory="examples", extensions=(".wav", ".mp3")):
    """Find audio files in the given directory"""
    audio_files = []
    for ext in extensions:
        audio_files.extend(list(Path(directory).glob(f"**/*{ext}")))
    return [str(f) for f in audio_files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Vietnamese ASR API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--audio-dir", default="examples", help="Directory with audio files")
    parser.add_argument("--audio-file", help="Specific audio file to test")
    parser.add_argument("--output", default="api_test_results.json", help="Output file for results")
    args = parser.parse_args()

    tester = ASRAPITester(api_url=args.url)

    if args.audio_file:
        # Test with specific file
        success = tester.test_transcribe(args.audio_file)
    else:
        # Find audio files in directory
        audio_files = find_audio_files(args.audio_dir)
        if not audio_files:
            logger.warning(f"No audio files found in {args.audio_dir}")
            # Use the script file itself as an invalid file for error testing
            tester.test_health()
            tester.test_models()
            tester.test_languages()
            tester.test_error_handling(__file__)
        else:
            success = tester.run_all_tests(audio_files)

    tester.save_results(args.output)
