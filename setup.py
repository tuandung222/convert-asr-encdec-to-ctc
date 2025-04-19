from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vietnamese-asr",
    version="0.1.0",
    author="Tuan Dung",
    author_email="tuandunghcmut@gmail.com",
    description="Vietnamese Automatic Speech Recognition with PhoWhisper-CTC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6.0",
        "torchaudio>=2.6.0",
        "transformers>=4.38.2",
        "huggingface_hub>=0.23.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "numpy>=1.26.4",
        "fastapi>=0.109.2",
        "uvicorn>=0.27.1",
        "pydantic>=2.6.1",
        "onnx>=1.15.0",
        "onnxruntime>=1.16.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.109.2",
            "uvicorn>=0.27.1",
            "python-multipart>=0.0.7",
            "prometheus-client",
            "opentelemetry-api",
            "opentelemetry-sdk",
            "psutil>=5.9.5",
        ],
        "ui": [
            "streamlit>=1.32.2",
            "pandas>=2.2.1",
            "plotly>=5.18.0",
            "audio-recorder-streamlit>=0.0.9",
            "pydub>=0.25.1",
            "python-dotenv>=1.0.0",
        ],
        "dev": [
            "pre-commit>=3.6.0",
            "black>=24.2.0",
            "isort>=5.13.2",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "bandit>=1.7.7",
        ],
    },
) 