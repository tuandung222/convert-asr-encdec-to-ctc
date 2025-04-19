import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodule import VietBudDataModule
from src.models.ctc_model import WhisperCTCModel

logger = logging.getLogger(__name__)


class TrainingManager:
    """Manages the training process for the WhisperCTC model."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.datamodule = None
        self.model = None
        self.trainer = None

        # Initialize components
        self._init_datamodule()
        self._init_model()
        self._init_trainer()

    def _init_datamodule(self) -> None:
        """Initialize the data module."""
        logger.info("Initializing data module")
        self.datamodule = VietBudDataModule(self.config)

    def _init_model(self) -> None:
        """Initialize the model."""
        logger.info("Initializing WhisperCTC model")
        self.model = WhisperCTCModel(self.config)

    def _init_trainer(self) -> None:
        """Initialize the PyTorch Lightning trainer."""
        logger.info("Initializing trainer")

        # Set up callbacks
        callbacks = self._get_callbacks()

        # Set up logger
        loggers = self._get_loggers()

        # Initialize trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            accelerator=self.config.training.accelerator,
            devices=self.config.training.devices,
            strategy=self.config.training.strategy,
            precision=self.config.training.precision,
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=self.config.training.log_every_n_steps,
            gradient_clip_val=self.config.training.gradient_clip_val,
            accumulate_grad_batches=self.config.training.accumulate_grad_batches,
        )

    def _get_callbacks(self) -> list:
        """Configure training callbacks."""
        callbacks = []

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=self.config.training.checkpointing.filename,
            save_top_k=self.config.training.checkpointing.save_top_k,
            monitor=self.config.training.checkpointing.monitor,
            mode=self.config.training.checkpointing.mode,
        )
        callbacks.append(checkpoint_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        # Early stopping (optional)
        early_stop_callback = EarlyStopping(
            monitor=self.config.training.checkpointing.monitor,
            mode=self.config.training.checkpointing.mode,
            patience=5,
            verbose=True,
        )
        callbacks.append(early_stop_callback)

        return callbacks

    def _get_loggers(self) -> list:
        """Configure training loggers."""
        loggers = []

        # WandB logger (if available)
        if "WANDB_API_KEY" in os.environ:
            wandb_logger = WandbLogger(
                project="vietnamese-asr-ctc",
                name=f"train-{self.config.model.name.split('/')[-1]}",
                log_model=True,
            )
            loggers.append(wandb_logger)

        return loggers if loggers else None

    def train(self) -> str:
        """Train the model and return the path to the best checkpoint."""
        logger.info("Starting training")

        # Prepare data
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="fit")

        # Train the model
        self.trainer.fit(self.model, datamodule=self.datamodule)

        # Get the best checkpoint path
        best_checkpoint_path = self.trainer.checkpoint_callback.best_model_path
        logger.info(f"Training completed. Best checkpoint: {best_checkpoint_path}")

        return best_checkpoint_path

    def test(self, checkpoint_path: str | None = None) -> dict:
        """Test the model and return the metrics."""
        logger.info("Starting testing")

        # Prepare test data
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="test")

        # Get checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.trainer.checkpoint_callback.best_model_path
            logger.info(f"Using best checkpoint: {checkpoint_path}")

        # Test the model
        test_results = self.trainer.test(
            self.model, datamodule=self.datamodule, ckpt_path=checkpoint_path
        )
        logger.info(f"Test results: {test_results}")

        return test_results[0] if test_results else {}
