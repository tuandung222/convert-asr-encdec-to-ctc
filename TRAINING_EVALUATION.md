# Training and Evaluation Process

This document provides a detailed explanation of the training and evaluation process for the Vietnamese Automatic Speech Recognition (ASR) system using a CTC-based approach derived from PhoWhisper.

## Overview

The training process involves converting a pre-trained PhoWhisper model from an encoder-decoder architecture to a CTC-based architecture. This conversion offers several advantages:

1. **Faster inference**: By eliminating the autoregressive decoder
2. **Simpler training**: No teacher forcing or complex decoding during training
3. **Reduced complexity**: Fewer parameters and simpler architecture
4. **Parallel prediction**: Predictions for all frames can be made simultaneously

## Data Preparation

### Dataset

The model is trained on the VietBud500 dataset, which contains Vietnamese speech recordings. For training efficiency, we use a subset of the dataset:

- 3 shards of the 105 training shards
- 1 shard of the test data (limited to 1000 examples)

```python
self.train_url = [
    "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/train-00000-of-00105-be5f872f8be772f5.parquet",
    "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/train-00097-of-00105-4160c0470220c086.parquet",
    "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/train-00086-of-00105-131a0bbf617d895c.parquet"
]
self.test_url = "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/test-00000-of-00002-531c1d81edb57297.parquet"
```

### Data Module

We use PyTorch Lightning's `LightningDataModule` for data handling. The `VietBud500DataModule` encapsulates:

1. **Data loading**: Loading and caching the dataset from HuggingFace
2. **Data preprocessing**: 
   - Audio feature extraction
   - Transcription tokenization
3. **Data splitting**: Creating train/validation/test splits
4. **Batch preparation**: Collating examples into batches

```python
class VietBud500DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        processor_name: str = "vinai/PhoWhisper-tiny",
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        # Initialization
        
    def prepare_data(self):
        # Download and prepare data
        
    def setup(self, stage=None):
        # Create train/val/test splits
        
    def collate_fn(self, batch):
        # Process audio and transcription
        # Extract features and tokenize transcriptions
        
    def train_dataloader(self):
        # Return DataLoader for training
        
    def val_dataloader(self):
        # Return DataLoader for validation
        
    def test_dataloader(self):
        # Return DataLoader for testing
```

### Feature Extraction

Audio features are extracted using the `WhisperProcessor`, which handles:

1. Audio normalization
2. Mel spectrogram computation
3. Feature normalization
4. Padding and batching

## Model Architecture

### Original PhoWhisper Architecture

The original PhoWhisper model follows an encoder-decoder architecture:

1. **Encoder**: Processes audio features to create contextual representations
2. **Decoder**: Autoregressively generates text tokens using encoder outputs

### CTC-Based Architecture

Our CTC-based adaptation:

1. **Encoder**: Reused from PhoWhisper (identical to original)
2. **CTC Head**: A simple linear layer mapping encoder outputs to vocabulary size

```python
class PhoWhisperCTCModel(nn.Module):
    def __init__(self, encoder, dim, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.ctc_head = torch.nn.Linear(dim, vocab_size)
    
    def forward(self, input_features, attention_mask=None):
        # Get encoder output
        encoder_out = self.encoder(input_features, attention_mask=attention_mask).last_hidden_state
        
        # Apply CTC head to get logits
        logits = self.ctc_head(encoder_out)
        
        return logits
```

### Lightning Module

We use PyTorch Lightning for training management through the `PhoWhisperLightningModule`:

```python
class PhoWhisperLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vinai/PhoWhisper-tiny",
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
    ):
        # Initialize module and load processor and config
        # Create encoder and CTC head
        # Set up CTC loss
        
    def forward(self, input_features, labels=None):
        # Forward pass through encoder
        # Forward pass through CTC head
        # Compute CTC loss if labels provided
        
    def training_step(self, batch, batch_idx):
        # Process batch and compute loss
        
    def validation_step(self, batch, batch_idx):
        # Validate model performance
        
    def test_step(self, batch, batch_idx):
        # Test model performance
        
    def configure_optimizers(self):
        # Configure AdamW optimizer
        # Configure learning rate scheduler
        
    def ctc_decode(self, logits, processor=None):
        # Decode CTC logits to text
        # Remove blanks and collapse duplicates
```

## Training Process

### Initialization

1. **Data preparation**:
   ```python
   datamodule = VietBud500DataModule(batch_size=24, processor_name="vinai/PhoWhisper-tiny")
   datamodule.prepare_data()
   datamodule.setup()
   ```

2. **Model initialization**:
   ```python
   lightning_module = PhoWhisperLightningModule(
       model_name="vinai/PhoWhisper-tiny", learning_rate=1e-4, warmup_steps=20
   )
   ```

3. **Callbacks setup**:
   ```python
   lr_monitor = LearningRateMonitor(logging_interval='step')
   checkpoint_callback = ModelCheckpoint(
       monitor='val_wer',
       mode='min',
       save_top_k=1,
       filename='best-{val_wer:.4f}'
   )
   eval_callback = EvalCallback(processor=datamodule.processor)
   ```

### Training Configuration

- **Batch size**: 24
- **Learning rate**: 1e-4
- **Warmup steps**: 20
- **Epochs**: 64
- **Precision**: bfloat16 mixed precision
- **Hardware**: GPU (single device)

```python
trainer = pl.Trainer(
    max_epochs=64,
    accelerator="gpu",
    devices=1,
    logger=True,
    precision="bf16-mixed",
    callbacks=[lr_monitor, checkpoint_callback, eval_callback],
)
```

### Training Execution

```python
trainer.fit(lightning_module, datamodule=datamodule)
```

During training:
1. **Forward pass**: Audio features are passed through the encoder, then the CTC head
2. **Loss calculation**: CTC loss is computed between logits and reference transcriptions
3. **Backward pass**: Gradients are computed and parameters updated
4. **Validation**: After each epoch, validation WER is computed
5. **Checkpointing**: Best model is saved based on validation WER

## CTC Training Details

### CTC Loss Calculation

The CTC loss is calculated as follows:

1. Encoder outputs are passed through the CTC head to get logits
2. Logits are transformed to log probabilities using softmax
3. CTC loss is calculated using log probabilities and reference transcriptions

```python
# Forward pass through encoder
encoder_outputs = self.encoder(input_features)  # (batch, time, hidden)
# Forward pass through CTC head
logits = self.ctc_head(encoder_outputs.last_hidden_state)  # (batch, time, vocab)
logits = logits.transpose(0, 1)  # (time, batch, vocab)

# Calculate log probabilities
log_probs = torch.nn.functional.log_softmax(logits, dim=2)

# Calculate input lengths
input_lengths = torch.full(
    size=(log_probs.size(1),),
    fill_value=log_probs.size(0),
    dtype=torch.int32,
)

# Handle labels (replace bos token with pad token as blank)
labels[labels == self.processor.tokenizer.bos_token_id] = (
    self.processor.tokenizer.pad_token_id
)

# Create label mask and calculate label lengths
label_mask = labels != self.processor.tokenizer.pad_token_id
labels = labels[label_mask].to(torch.int32)
label_lengths = label_mask.sum(dim=1)

# Calculate CTC loss
loss = self.ctc_loss(log_probs, labels, input_lengths, label_lengths)
```

### Optimizer Configuration

We use the AdamW optimizer with cosine learning rate scheduling:

```python
optimizer = AdamW(
    self.parameters(),
    lr=self.learning_rate,
    weight_decay=0.1,
    betas=(0.9, 0.98),
    eps=1e-6,
)

# Calculate total steps
total_steps = len(train_dataloader) * self.trainer.max_epochs

# Create learning rate scheduler
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=self.warmup_steps,
    num_training_steps=total_steps,
)
```

## Evaluation Process

### Metrics

We use Word Error Rate (WER) as the primary evaluation metric, which measures the edit distance between predicted and reference transcriptions at the word level.

```python
wer_metric = load_metric("wer")
wer = wer_metric.compute(predictions=predictions, references=references)
```

### Evaluation During Training

During training, an `EvalCallback` tracks validation performance:

```python
class EvalCallback(pl.Callback):
    def __init__(self, processor):
        # Initialize callback with processor
        # Set up prediction and reference collection
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Collect predictions and references
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Calculate WER
        # Log results
        # Display examples
```

The callback:
1. Collects predicted and reference texts from validation batches
2. Calculates WER at the end of each validation epoch
3. Logs the WER and displays example predictions

### CTC Decoding

For inference, we decode the CTC output as follows:

```python
def ctc_decode(self, logits, processor=None):
    # Convert logits to class indices
    class_indices = logits.argmax(dim=2)
    
    texts = []
    for seq in class_indices:
        # Remove blanks (pad tokens)
        seq_no_blank = seq[seq != processor.tokenizer.pad_token_id]
        
        # Collapse repeats
        seq_collapsed = []
        prev_token = -1
        for token in seq_no_blank:
            if token != prev_token:
                seq_collapsed.append(token.item())
                prev_token = token
        
        # Decode to text
        text = processor.decode(seq_collapsed, skip_special_tokens=False)
        texts.append(text)
    
    return texts
```

The decoding process:
1. Takes the argmax of logits to get token indices
2. Removes blank tokens (pad tokens)
3. Collapses repeated tokens
4. Decodes indices to text using the tokenizer

### Final Evaluation

After training, final evaluation is performed on the test set:

```python
trainer.test(lightning_module, datamodule=datamodule, ckpt_path='best')
```

This process:
1. Loads the best checkpoint based on validation WER
2. Evaluates the model on the test set
3. Reports final test WER

## Performance Results

The trained model achieves:

- **Word Error Rate (WER)**: ~41% on the test set
- **Real-time factor (RTF)**: <0.5x (more than 2x faster than real-time)
- **Memory usage**: <500MB

## Inference Implementation

The final model is implemented in `src/models/inference_model.py` with:

1. **Model loading** from HuggingFace Hub
2. **Audio preprocessing** using torchaudio
3. **Feature extraction** using WhisperFeatureExtractor
4. **Inference pipeline** with CPU or GPU support
5. **CTC decoding** for final transcription

## Conclusion

The training and evaluation process successfully converts PhoWhisper from an encoder-decoder to a CTC-based architecture, resulting in:

1. **Faster inference**: More than 2x faster than real-time
2. **Reasonable accuracy**: ~41% WER on Vietnamese speech
3. **Reduced complexity**: Smaller model footprint and simpler architecture
4. **Deployment efficiency**: Suitable for CPU deployment with low memory requirements

This approach demonstrates that CTC-based models can offer an excellent trade-off between performance and efficiency for ASR tasks, particularly in resource-constrained environments. 