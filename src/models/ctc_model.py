from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import (
    WhisperEncoder,
    AutoConfig,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from bitsandbytes.optim import Adam8bit
import hydra
from omegaconf import DictConfig
import evaluate
from dataclasses import dataclass


@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class CTCHead(nn.Module):
    """Custom CTC head for ASR."""
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        use_gelu: bool = True,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.GELU() if use_gelu else nn.ReLU(),
                nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity(),
            ])
        layers.append(nn.Linear(hidden_size, vocab_size))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden_states)


class WhisperCTCModel(pl.LightningModule):
    """PhoWhisper model modified to use CTC loss instead of encoder-decoder architecture."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Load pretrained components
        self.processor = AutoProcessor.from_pretrained(config.model.name)
        whisper_config = AutoConfig.from_pretrained(config.model.name)
        
        # Initialize encoder from pretrained
        self.encoder = WhisperEncoder(config=whisper_config)
        
        # Initialize CTC head
        self.ctc_head = CTCHead(
            hidden_size=config.model.ctc_head.hidden_size,
            vocab_size=self.processor.tokenizer.vocab_size,
            num_layers=config.model.ctc_head.num_layers,
            use_layer_norm=config.model.ctc_head.use_layer_norm,
            use_gelu=config.model.ctc_head.use_gelu,
        )
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(
            blank=self.processor.tokenizer.pad_token_id,
            zero_infinity=True,
        )
        
        # Metrics
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        
    def forward(
        self, input_features: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        # Encode audio features
        encoder_outputs = self.encoder(input_features)
        
        # Get logits from CTC head
        logits = self.ctc_head(encoder_outputs.last_hidden_state)
        logits = logits.transpose(0, 1)  # (time, batch, vocab)
        
        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        
        # Prepare lengths for CTC loss
        input_lengths = torch.full(
            size=(log_probs.size(1),),
            fill_value=log_probs.size(0),
            dtype=torch.int32,
            device=log_probs.device,
        )
        
        if labels is not None:
            # Replace BOS token with pad token (blank token)
            labels = labels.clone()
            labels[labels == self.processor.tokenizer.bos_token_id] = (
                self.processor.tokenizer.pad_token_id
            )
            
            # Get valid label lengths
            label_mask = labels != self.processor.tokenizer.pad_token_id
            valid_labels = labels[label_mask].to(torch.int32)
            label_lengths = label_mask.sum(dim=1)
            
            # Calculate CTC loss
            loss = self.ctc_loss(log_probs, valid_labels, input_lengths, label_lengths)
            
            return ModelOutput(loss=loss, logits=logits)
        
        return ModelOutput(logits=logits)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch["input_features"], batch["labels"])
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch["input_features"], batch["labels"])
        self.log("val_loss", outputs.loss, prog_bar=True)
        
        # Compute WER and CER
        predictions = self.decode(outputs.logits)
        references = self.processor.batch_decode(batch["labels"], skip_special_tokens=True)
        
        wer = self.wer_metric.compute(predictions=predictions, references=references)
        cer = self.cer_metric.compute(predictions=predictions, references=references)
        
        self.log("val_wer", wer, prog_bar=True)
        self.log("val_cer", cer, prog_bar=True)
    
    def decode(self, logits: torch.Tensor) -> list[str]:
        """Decode logits to text using greedy decoding."""
        predictions = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predictions.transpose(0, 1))
    
    def configure_optimizers(self):
        # Initialize optimizer
        optimizer = Adam8bit(
            self.parameters(),
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
            eps=self.hparams.optimizer.eps,
            betas=self.hparams.optimizer.betas,
        )
        
        # Initialize scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.scheduler.num_warmup_steps,
            num_training_steps=self.hparams.scheduler.num_training_steps,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        } 