# Vietnamese Speech Recognition with CTC

This project implements an Automatic Speech Recognition (ASR) system for Vietnamese using CTC (Connectionist Temporal Classification). The model is based on a modified version of PhoWhisper, converting its encoder-decoder architecture to a CTC-based architecture for more efficient training and inference.

## Project Structure

```
speech_processing/
├── configs/           # Configuration files for model and training
│   ├── training_config.yaml     # Training configuration
│   ├── inference_config.yaml    # Inference configuration
│   └── model_config.yaml        # Model architecture configuration
├── source/            # Source code
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model architecture definitions
│   ├── training/      # Training utilities
│   └── utils/         # Helper functions
├── scripts/           # Training and inference scripts
│   ├── train.py       # Training script
│   ├── inference.py   # Inference script for batch processing
│   └── app.py         # Gradio app for interactive demo
├── notebooks/         # Jupyter notebooks for exploration
├── run.py             # Main entry point
└── tests/             # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/speech_processing.git
cd speech_processing
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

The project provides a single entry point (`run.py`) for all operations:

### Training:
```bash
python run.py train
```

### Inference:
```bash
python run.py infer --audio /path/to/audio_file_or_directory
```

### Interactive Demo:
```bash
python run.py app
```

## Model Architecture

The model uses the encoder from PhoWhisper and replaces the decoder with a CTC head. This modification:
- Simplifies the architecture
- Enables faster training and inference
- Maintains competitive accuracy

Key components:
- PhoWhisper encoder for feature extraction
- Custom CTC head for Vietnamese ASR
- Efficient data loading with PyTorch Lightning

## Training

1. Configure your training parameters in `configs/training_config.yaml`
2. Run training:
```bash
python run.py train
```

Training features:
- Mixed precision training
- Multi-GPU support
- Wandb logging
- Checkpointing

## CPU Inference

This model is optimized for CPU inference, making it suitable for deployment in environments without GPUs.

### Option 1: Direct Script Inference

For batch processing of audio files:

```bash
python run.py infer --audio /path/to/audio_file_or_directory --device cpu
```

Parameters:
- `--audio`: Path to an audio file or directory containing audio files
- `--output`: Path to save transcription results (default: outputs/transcriptions.txt)
- `--checkpoint`: Path to model checkpoint (optional)
- `--device`: Device to run inference on (default: cpu)

### Option 2: Interactive Demo with Gradio

For interactive testing with a web interface:

```bash
python run.py app --device cpu
```

Parameters:
- `--checkpoint`: Path to model checkpoint (optional)
- `--device`: Device to run inference on (default: cpu)
- `--share`: Share the app publicly through Gradio
- `--port`: Port to run the app on (default: 7860)

### Download Pre-trained Model

You can download the pre-trained model from HuggingFace:

```python
from huggingface_hub import hf_hub_download

# Download the checkpoint
checkpoint_path = hf_hub_download(
    repo_id="tuandunghcmut/PhoWhisper-tiny-CTC",
    filename="best-val_wer=0.3986.ckpt",
    local_dir="./checkpoints",
)
```

Then use it for inference:

```bash
python run.py infer --audio /path/to/audio --checkpoint ./checkpoints/best-val_wer=0.3986.ckpt --device cpu
```

## CPU Performance

On a standard CPU, the model achieves:
- Real-time factor (RTF): ~0.3-0.5x (2-3x faster than real-time)
- Memory usage: < 500MB
- Minimal latency for short audio clips

## Evaluation

The model achieves competitive results on Vietnamese speech recognition:
- WER (Word Error Rate): 41% on the VietBud500 test set
- CER (Character Error Rate): Comparable to state-of-the-art models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{vietnamese_asr_ctc,
  author = {Dung Vo Pham Tuan},
  title = {Vietnamese Speech Recognition with CTC},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc}
}
```

## Acknowledgments

- VINAI for the PhoWhisper model
- VietBud500 dataset creators 