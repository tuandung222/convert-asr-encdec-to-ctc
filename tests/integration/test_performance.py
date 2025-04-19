#!/usr/bin/env python

"""
Performance test script for the Vietnamese ASR FastAPI server
This script measures response times, throughput, and concurrent request handling
"""

import argparse
import concurrent.futures
import json
import logging
import os
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ASRPerformanceTester:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.results = {
            "single_requests": [],
            "batch_results": [],
            "concurrent_results": [],
            "summary": {},
        }

    def test_single_request(
        self, audio_file, model="phowhisper-tiny-ctc", language="vi", repeats=5
    ):
        """Test single request performance by repeating the same request multiple times"""
        logger.info(f"Testing single request performance with {audio_file}, {repeats} times")

        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return False

        response_times = []
        real_time_factors = []

        for i in range(repeats):
            logger.info(f"Request {i+1}/{repeats}")
            try:
                with open(audio_file, "rb") as f:
                    files = {"file": (os.path.basename(audio_file), f, "audio/wav")}
                    data = {"model": model, "language": language}

                    start_time = time.time()
                    response = requests.post(f"{self.api_url}/transcribe", files=files, data=data)
                    response.raise_for_status()
                    end_time = time.time()

                    response_time = end_time - start_time
                    result = response.json()

                    response_times.append(response_time)
                    real_time_factors.append(response_time / result["duration"])

                    logger.info(
                        f"Response time: {response_time:.2f}s, RTF: {response_time / result['duration']:.2f}"
                    )

                    self.results["single_requests"].append(
                        {
                            "request_id": i,
                            "audio_file": audio_file,
                            "response_time": response_time,
                            "audio_duration": result["duration"],
                            "real_time_factor": response_time / result["duration"],
                            "transcription": result["text"],
                        }
                    )

                # Wait a bit between requests to avoid overloading the server
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                self.results["single_requests"].append(
                    {"request_id": i, "audio_file": audio_file, "error": str(e)}
                )

        # Calculate statistics
        if response_times:
            stats = {
                "audio_file": audio_file,
                "mean_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "stddev_response_time": (
                    statistics.stdev(response_times) if len(response_times) > 1 else 0
                ),
                "mean_rtf": statistics.mean(real_time_factors),
                "median_rtf": statistics.median(real_time_factors),
                "min_rtf": min(real_time_factors),
                "max_rtf": max(real_time_factors),
            }

            self.results["summary"]["single_request"] = stats

            logger.info(
                f"Single request stats: Mean response time: {stats['mean_response_time']:.2f}s, Mean RTF: {stats['mean_rtf']:.2f}"
            )
            return True
        else:
            logger.error("No successful requests")
            return False

    def test_concurrent_requests(
        self, audio_file, model="phowhisper-tiny-ctc", language="vi", num_concurrent=5, repeats=3
    ):
        """Test handling of concurrent requests"""
        logger.info(
            f"Testing concurrent request performance with {audio_file}, {num_concurrent} concurrent requests, {repeats} times"
        )

        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return False

        all_response_times = []

        for iteration in range(repeats):
            logger.info(f"Iteration {iteration+1}/{repeats}")

            # Define the worker function for a single request
            def worker(worker_id):
                try:
                    with open(audio_file, "rb") as f:
                        files = {"file": (os.path.basename(audio_file), f, "audio/wav")}
                        data = {"model": model, "language": language}

                        start_time = time.time()
                        response = requests.post(
                            f"{self.api_url}/transcribe", files=files, data=data
                        )
                        response.raise_for_status()
                        end_time = time.time()

                        response_time = end_time - start_time
                        result = response.json()

                        return {
                            "worker_id": worker_id,
                            "iteration": iteration,
                            "response_time": response_time,
                            "audio_duration": result["duration"],
                            "real_time_factor": response_time / result["duration"],
                            "success": True,
                        }
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed: {str(e)}")
                    return {
                        "worker_id": worker_id,
                        "iteration": iteration,
                        "error": str(e),
                        "success": False,
                    }

            # Start concurrent workers
            worker_results = []
            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(worker, i) for i in range(num_concurrent)]
                for future in concurrent.futures.as_completed(futures):
                    worker_results.append(future.result())

            end_time = time.time()
            total_time = end_time - start_time

            # Calculate statistics for this iteration
            successful_results = [r for r in worker_results if r.get("success", False)]
            response_times = [r["response_time"] for r in successful_results]

            if response_times:
                all_response_times.extend(response_times)

                iteration_result = {
                    "iteration": iteration,
                    "total_time": total_time,
                    "num_concurrent": num_concurrent,
                    "num_successful": len(successful_results),
                    "mean_response_time": statistics.mean(response_times),
                    "max_response_time": max(response_times),
                    "min_response_time": min(response_times),
                    "throughput": len(successful_results) / total_time,
                    "worker_results": worker_results,
                }

                self.results["concurrent_results"].append(iteration_result)

                logger.info(
                    f"Iteration {iteration+1} complete: {len(successful_results)}/{num_concurrent} successful, "
                    + f"Mean response time: {iteration_result['mean_response_time']:.2f}s, "
                    + f"Throughput: {iteration_result['throughput']:.2f} req/s"
                )

            # Wait between iterations to let the server recover
            time.sleep(2)

        # Calculate overall statistics
        if all_response_times:
            stats = {
                "audio_file": audio_file,
                "num_concurrent": num_concurrent,
                "repeats": repeats,
                "mean_response_time": statistics.mean(all_response_times),
                "median_response_time": statistics.median(all_response_times),
                "min_response_time": min(all_response_times),
                "max_response_time": max(all_response_times),
                "stddev_response_time": (
                    statistics.stdev(all_response_times) if len(all_response_times) > 1 else 0
                ),
            }

            self.results["summary"]["concurrent_requests"] = stats

            logger.info(
                f"Concurrent request stats: Mean response time: {stats['mean_response_time']:.2f}s"
            )
            return True
        else:
            logger.error("No successful concurrent requests")
            return False

    def save_results(self, output_file="performance_results.json"):
        """Save test results to a file"""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Test results saved to {output_file}")

    def generate_graphs(self, output_dir="performance_results"):
        """Generate graphs based on the test results"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Single request response times
        if self.results.get("single_requests"):
            plt.figure(figsize=(10, 6))
            response_times = [
                r["response_time"] for r in self.results["single_requests"] if "response_time" in r
            ]

            if response_times:
                plt.plot(range(1, len(response_times) + 1), response_times, "o-")
                plt.xlabel("Request Number")
                plt.ylabel("Response Time (s)")
                plt.title("Single Request Response Times")
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "single_request_times.png"))
                plt.close()

        # Concurrent request response times
        if self.results.get("concurrent_results"):
            plt.figure(figsize=(10, 6))
            iterations = []
            mean_times = []

            for r in self.results["concurrent_results"]:
                iterations.append(r["iteration"] + 1)
                mean_times.append(r["mean_response_time"])

            if iterations and mean_times:
                plt.bar(iterations, mean_times)
                plt.xlabel("Iteration")
                plt.ylabel("Mean Response Time (s)")
                plt.title("Mean Response Time per Concurrent Batch")
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "concurrent_mean_times.png"))
                plt.close()

                # Throughput graph
                plt.figure(figsize=(10, 6))
                throughputs = [r["throughput"] for r in self.results["concurrent_results"]]
                plt.bar(iterations, throughputs)
                plt.xlabel("Iteration")
                plt.ylabel("Throughput (req/s)")
                plt.title("Throughput per Concurrent Batch")
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "concurrent_throughput.png"))
                plt.close()

        # Create a summary plot if we have both types of tests
        if self.results.get("summary", {}).get("single_request") and self.results.get(
            "summary", {}
        ).get("concurrent_requests"):
            single_mean = self.results["summary"]["single_request"]["mean_response_time"]
            concurrent_mean = self.results["summary"]["concurrent_requests"]["mean_response_time"]

            plt.figure(figsize=(10, 6))
            labels = ["Single Requests", "Concurrent Requests"]
            means = [single_mean, concurrent_mean]

            plt.bar(labels, means)
            plt.ylabel("Mean Response Time (s)")
            plt.title("Response Time Comparison")
            plt.grid(True)

            # Add values on top of bars
            for i, v in enumerate(means):
                plt.text(i, v + 0.1, f"{v:.2f}s", ha="center")

            plt.savefig(os.path.join(output_dir, "response_time_comparison.png"))
            plt.close()

        logger.info(f"Graphs saved to {output_dir} directory")


def find_audio_files(directory="examples", extensions=(".wav", ".mp3")):
    """Find audio files in the given directory"""
    audio_files = []
    for ext in extensions:
        audio_files.extend(list(Path(directory).glob(f"**/*{ext}")))
    return [str(f) for f in audio_files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance test for the Vietnamese ASR API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--audio-file", help="Specific audio file to test")
    parser.add_argument("--audio-dir", default="examples", help="Directory with audio files")
    parser.add_argument(
        "--output", default="performance_results.json", help="Output file for results"
    )
    parser.add_argument(
        "--graphs-dir", default="performance_results", help="Directory for output graphs"
    )
    parser.add_argument(
        "--single-repeats", type=int, default=5, help="Number of repeats for single request test"
    )
    parser.add_argument("--concurrent", type=int, default=3, help="Number of concurrent requests")
    parser.add_argument(
        "--concurrent-repeats", type=int, default=3, help="Number of repeats for concurrent test"
    )
    parser.add_argument("--skip-single", action="store_true", help="Skip single request tests")
    parser.add_argument(
        "--skip-concurrent", action="store_true", help="Skip concurrent request tests"
    )
    args = parser.parse_args()

    tester = ASRPerformanceTester(api_url=args.url)

    audio_file = args.audio_file
    if not audio_file:
        # Find an audio file to use for testing
        audio_files = find_audio_files(args.audio_dir)
        if audio_files:
            audio_file = audio_files[0]
            logger.info(f"Using audio file: {audio_file}")
        else:
            logger.error(f"No audio files found in {args.audio_dir}")
            exit(1)

    if not args.skip_single:
        tester.test_single_request(audio_file, repeats=args.single_repeats)

    if not args.skip_concurrent:
        tester.test_concurrent_requests(
            audio_file, num_concurrent=args.concurrent, repeats=args.concurrent_repeats
        )

    tester.save_results(args.output)
    tester.generate_graphs(args.graphs_dir)
