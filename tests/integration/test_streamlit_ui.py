#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Local test script for the Vietnamese ASR Streamlit UI
This script uses Selenium to test the Streamlit UI
"""

import os
import time
import json
import argparse
import logging
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class StreamlitUITester:
    def __init__(self, streamlit_url="http://localhost:8501", headless=True):
        self.streamlit_url = streamlit_url
        self.headless = headless
        self.driver = None
        self.results = {
            "ui_load": None,
            "upload_tests": [],
            "errors": []
        }
    
    def setup_driver(self):
        """Set up the Selenium WebDriver"""
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless")
        
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        try:
            self.driver = webdriver.Chrome(options=options)
            logger.info("WebDriver initialized successfully")
            return True
        except WebDriverException as e:
            logger.error(f"Failed to initialize WebDriver: {str(e)}")
            self.results["errors"].append({
                "stage": "setup",
                "error": str(e)
            })
            return False
    
    def close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")
    
    def test_ui_load(self):
        """Test if the Streamlit UI loads correctly"""
        logger.info(f"Testing UI load at {self.streamlit_url}")
        try:
            self.driver.get(self.streamlit_url)
            
            # Wait for Streamlit to initialize
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "h1"))
            )
            
            # Get the title
            title = self.driver.title
            logger.info(f"UI loaded successfully. Title: {title}")
            
            self.results["ui_load"] = {
                "success": True,
                "title": title
            }
            return True
        except TimeoutException:
            logger.error("Timeout waiting for UI to load")
            self.results["ui_load"] = {
                "success": False,
                "error": "Timeout waiting for UI to load"
            }
            self.results["errors"].append({
                "stage": "ui_load",
                "error": "Timeout waiting for UI to load"
            })
            return False
        except Exception as e:
            logger.error(f"Failed to load UI: {str(e)}")
            self.results["ui_load"] = {
                "success": False,
                "error": str(e)
            }
            self.results["errors"].append({
                "stage": "ui_load",
                "error": str(e)
            })
            return False
    
    def test_file_upload(self, audio_file):
        """Test file upload functionality"""
        logger.info(f"Testing file upload with {audio_file}")
        
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            self.results["errors"].append({
                "stage": "file_upload",
                "error": f"File not found: {audio_file}"
            })
            return False
        
        try:
            # Wait for upload button to be available
            upload_button = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Upload') or contains(text(), 'Browse')]"))
            )
            
            # Click upload button
            upload_button.click()
            logger.info("Clicked upload button")
            
            # Wait for file input to be available
            file_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
            )
            
            # Send the file path to the input
            file_input.send_keys(os.path.abspath(audio_file))
            logger.info("Sent file path to input")
            
            # Wait for transcription to complete (looking for text area or result div)
            try:
                result_element = WebDriverWait(self.driver, 60).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'stText') and contains(text(), '')]"))
                )
                transcription_text = result_element.text
                logger.info(f"Transcription found: {transcription_text}")
                
                self.results["upload_tests"].append({
                    "audio_file": audio_file,
                    "success": True,
                    "transcription": transcription_text
                })
                return True
            except TimeoutException:
                logger.warning("Timeout waiting for transcription result")
                self.results["upload_tests"].append({
                    "audio_file": audio_file,
                    "success": False,
                    "error": "Timeout waiting for transcription result"
                })
                return False
        except Exception as e:
            logger.error(f"File upload test failed: {str(e)}")
            self.results["upload_tests"].append({
                "audio_file": audio_file,
                "success": False,
                "error": str(e)
            })
            self.results["errors"].append({
                "stage": "file_upload",
                "error": str(e),
                "audio_file": audio_file
            })
            return False
    
    def run_all_tests(self, audio_files):
        """Run all UI tests in sequence"""
        logger.info("Running all UI tests...")
        
        if not self.setup_driver():
            logger.error("Failed to set up WebDriver, stopping tests")
            return False
        
        try:
            if not self.test_ui_load():
                logger.error("UI failed to load, stopping tests")
                return False
            
            for audio_file in audio_files:
                self.test_file_upload(audio_file)
                # Reload the page for the next test
                self.driver.get(self.streamlit_url)
                time.sleep(2)  # Wait for the page to reload
            
            return len(self.results["errors"]) == 0
        finally:
            self.close_driver()
    
    def save_results(self, output_file="ui_test_results.json"):
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
    parser = argparse.ArgumentParser(description="Test the Vietnamese ASR Streamlit UI")
    parser.add_argument("--url", default="http://localhost:8501", help="Streamlit UI URL")
    parser.add_argument("--audio-dir", default="examples", help="Directory with audio files")
    parser.add_argument("--audio-file", help="Specific audio file to test")
    parser.add_argument("--output", default="ui_test_results.json", help="Output file for results")
    parser.add_argument("--no-headless", action="store_true", help="Run browser in non-headless mode")
    args = parser.parse_args()
    
    tester = StreamlitUITester(
        streamlit_url=args.url,
        headless=not args.no_headless
    )
    
    if args.audio_file:
        # Test with specific file
        if tester.setup_driver():
            try:
                tester.test_ui_load()
                tester.test_file_upload(args.audio_file)
            finally:
                tester.close_driver()
    else:
        # Find audio files in directory
        audio_files = find_audio_files(args.audio_dir)
        if not audio_files:
            logger.warning(f"No audio files found in {args.audio_dir}")
            if tester.setup_driver():
                try:
                    tester.test_ui_load()
                finally:
                    tester.close_driver()
        else:
            tester.run_all_tests(audio_files[:3])  # Test with up to 3 files
    
    tester.save_results(args.output) 