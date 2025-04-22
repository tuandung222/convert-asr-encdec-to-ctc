import argparse
import json
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.datamodule import VietBudDataModule
from src.models.ctc_model import WhisperCTCModel


def evaluate_model(model: WhisperCTCModel, dataloader: torch.utils.data.DataLoader) -> dict:
    """Evaluate model on a dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_references = []
    all_wer = []
    all_cer = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Get model predictions
            outputs = model(batch["input_features"])
            predictions = model.decode(outputs.logits)
            references = model.processor.batch_decode(batch["labels"], skip_special_tokens=True)

            # Calculate metrics
            wer = model.wer_metric.compute(predictions=predictions, references=references)
            cer = model.cer_metric.compute(predictions=predictions, references=references)

            # Store results
            all_predictions.extend(predictions)
            all_references.extend(references)
            all_wer.append(wer)
            all_cer.append(cer)

    # Calculate average metrics
    avg_wer = sum(all_wer) / len(all_wer)
    avg_cer = sum(all_cer) / len(all_cer)

    return {
        "predictions": all_predictions,
        "references": all_references,
        "wer": avg_wer,
        "cer": avg_cer,
    }


@hydra.main(config_path="../configs", config_name="training_config")
def main(config: DictConfig) -> None:
    """Main evaluation function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model from checkpoint
    model = WhisperCTCModel.load_from_checkpoint(
        args.checkpoint,
        config=config,
    )

    # Initialize data module
    datamodule = VietBudDataModule(config)
    datamodule.setup("test")

    # Evaluate model
    results = evaluate_model(model, datamodule.test_dataloader())

    # Save metrics
    metrics = {
        "wer": results["wer"],
        "cer": results["cer"],
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "reference": results["references"],
            "prediction": results["predictions"],
        }
    )
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    print(f"Evaluation results saved to {output_dir}")
    print(f"Word Error Rate: {results['wer']:.4f}")
    print(f"Character Error Rate: {results['cer']:.4f}")


if __name__ == "__main__":
    main()
