#!/usr/bin/env python

# - This notebook is used for training the model
# - Please refer to my notebook named "evaluate_after_training" for:
#     - Loading the best checkpoint from my Huggingface repository (I have pushed to it after training)
#     - Evaluating the model on the test set
#     - Visualizing the results (if you want to hear some audio from the test set and see the predicted transcription from the model)

# # Define Datamodule, ModelModule, and Eval Utils

# In[6]:


# %pip install datasets transformers librosa soundfile jiwer evaluate --quiet
# %pip install pytorch_lightning torch bitsandbytes --quiet


# In[7]:


import bitsandbytes as bnb
import pytorch_lightning as pl
import torch
from bitsandbytes.optim import Adam8bit
from datasets import load_dataset
from easydict import EasyDict as edict
from evaluate import load as load_metric
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from transformers.models.whisper.modeling_whisper import WhisperEncoder


class VietBud500DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        processor_name: str = "vinai/PhoWhisper-tiny",
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(processor_name)

        print("Download just 3 shards / 105 shards of the origin training data")
        self.train_url = [
            "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/train-00000-of-00105-be5f872f8be772f5.parquet",
            "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/train-00097-of-00105-4160c0470220c086.parquet",
            "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/train-00086-of-00105-131a0bbf617d895c.parquet",
        ]
        self.test_url = "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/test-00000-of-00002-531c1d81edb57297.parquet"
        self.data_files = {"train": self.train_url, "test": self.test_url}

        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        self.dataset = load_dataset(
            "parquet",
            data_files=self.data_files,
        )
        self.sampling_rate = self.dataset["train"].features["audio"].sampling_rate

    def setup(self, stage=None):
        test_dataset = self.dataset["test"]

        train_dataset = self.dataset["train"].shuffle(seed=42)
        train_val_split = train_dataset.train_test_split(test_size=0.05, seed=42)
        self.train_dataset = train_val_split["train"]
        self.val_dataset = train_val_split["test"]

        print(
            "Just select 1000 examples from a shard of the origin test data serving as the test split!"
        )
        self.test_dataset = test_dataset.select(range(1000))

        print("Number of training examples:", len(self.train_dataset))
        print("Number of validation examples:", len(self.val_dataset))
        print("Number of test examples:", len(self.test_dataset))

    def collate_fn(self, batch):
        # Extract audio and transcription from the batch
        audios = [item["audio"]["array"] for item in batch]
        transcriptions = [item["transcription"] for item in batch]

        # Process audio and transcription using the processor
        inputs = self.processor(
            audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )

        # Tokenize transcriptions
        # with self.processor.as_target_processor(): # prepared correctly without the decoder's bos token
        # if True:
        labels = self.processor(
            text=transcriptions,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).input_ids

        return {
            "input_features": inputs.input_features,
            "labels": labels,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )


# In[8]:


class PhoWhisperLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vinai/PhoWhisper-tiny",
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

        # self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

        temp_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.encoder = WhisperEncoder(config=self.config)
        self.encoder.load_state_dict(temp_model.model.encoder.state_dict(), strict=True)
        del temp_model

        self.ctc_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.processor.tokenizer.vocab_size),
        )

        self.ctc_loss = torch.nn.CTCLoss(
            blank=self.processor.tokenizer.pad_token_id, zero_infinity=True
        )

        # Hyperparameters for AdamW optimizer
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

    def forward(self, input_features, labels=None):
        encoder_outputs = self.encoder(input_features)  # (batch, time, hidden)
        logits = self.ctc_head(encoder_outputs.last_hidden_state)  # (batch, time, vocab)
        logits = logits.transpose(0, 1)  # (time, batch, vocab)

        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        input_lengths = torch.full(
            size=(log_probs.size(1),),
            fill_value=log_probs.size(0),
            dtype=torch.int32,
        )
        if labels is not None:

            # replace first bos token by pad token (blank token)
            labels[labels == self.processor.tokenizer.bos_token_id] = (
                self.processor.tokenizer.pad_token_id
            )

            label_mask = labels != self.processor.tokenizer.pad_token_id
            labels = labels[label_mask].to(torch.int32)
            label_lengths = label_mask.sum(dim=1)
            assert label_lengths.sum() == labels.size(
                0
            )  # "Sum of label_lengths must equal number of labels."

            loss = self.ctc_loss(log_probs, labels, input_lengths, label_lengths)
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return edict(
                {
                    "loss": loss,
                    "logits": logits if labels is not None else None,
                }
            )
        else:
            return edict({"logits": logits})

    def seq2seq_forward(self, input_features, labels=None):
        # Seq2Seq forward pass
        # Current not used
        return self.model(input_features=input_features, labels=labels)

    def training_step(self, batch, batch_idx):
        input_features = batch["input_features"]

        labels = batch["labels"]

        outputs = self(input_features=input_features, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        labels = batch["labels"]
        outputs = self(input_features=input_features, labels=labels)
        loss = outputs.loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        labels = batch["labels"]
        outputs = self(input_features=input_features, labels=labels)
        loss = outputs.loss
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return outputs

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
        # optimizer = Adam8bit(self.parameters(), lr=self.learning_rate, eps=1e-8)
        train_dataloader = self.trainer.datamodule.train_dataloader()
        total_steps = len(train_dataloader) * self.trainer.max_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint):
        # Save the processor along with the model checkpoint
        checkpoint["processor"] = self.processor

    def on_load_checkpoint(self, checkpoint):
        self.processor = checkpoint["processor"]

    def ctc_decode(self, logits, processor=None):
        if processor is None:
            processor = self.processor
        # logits shape: (time, batch, vocab)
        logits = logits.transpose(0, 1)  # (batch, time, vocab)
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


# In[9]:


def wer_evaluate(pl_module, test_dataloader, device="cuda"):
    # This legacy function is uses for Seq2Seq model, currently not used
    # Load the WER metric
    wer_metric = load_metric("wer")
    # Initialize lists to hold predictions and references
    predictions = []
    references = []
    pl_module.to(device)
    # Set the model to evaluation mode
    pl_module.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                # Move input features and labels to the correct device
                input_features = batch["input_features"].to(pl_module.device)
                labels = batch["labels"].to(pl_module.device)

                # Generate outputs
                outputs = pl_module.model.generate(input_features=input_features, do_sample=True)
                # Decode generated outputs to text
                predicted_texts = datamodule.processor.batch_decode(
                    outputs, skip_special_tokens=True
                )
                # Handle labels: replace -100 with pad_token_id and decode
                labels_cpu = labels.detach().cpu()
                label_texts = datamodule.processor.batch_decode(
                    labels_cpu, skip_special_tokens=True
                )
                # Collect predictions and references
                predictions.extend(predicted_texts)
                references.extend(label_texts)
    # Compute WER
    wer = wer_metric.compute(predictions=predictions, references=references)
    # Return the results as a dictionary
    return {"wer": wer}


def wer_ctc_evaluate(pl_module, test_dataloader, device="cuda"):
    wer_metric = load_metric("wer")

    predictions = []
    references = []
    pl_module.to(device)
    pl_module.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_features = batch["input_features"].to(pl_module.device)
                labels = batch["labels"].to(pl_module.device)
                logits = pl_module(input_features=input_features, labels=None).logits

                predicted_texts = pl_module.ctc_decode(logits)
                label_texts = pl_module.processor.batch_decode(labels, skip_special_tokens=True)

                predictions.extend(predicted_texts)
                references.extend(label_texts)

    wer = wer_metric.compute(predictions=predictions, references=references)
    print("First 5 predictions: ", predictions[:5])
    print("First 5 references: ", references[:5])
    print("WER:", wer)
    return {"wer": wer}


class EvalCallback(pl.Callback):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.val_predicted_texts = []
        self.val_reference_texts = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Collect predicted texts and reference texts from the validation batch
        logits = outputs.logits.detach().cpu()
        labels = batch["labels"].detach().cpu()
        # Decode logits to predicted texts
        predicted_texts = self.ctc_decode(logits, self.processor)
        # Decode labels to reference texts
        reference_texts = self.processor.batch_decode(labels, skip_special_tokens=True)
        # Collect them
        self.val_predicted_texts.extend(predicted_texts)
        self.val_reference_texts.extend(reference_texts)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Compute WER
        wer_metric = load_metric("wer")
        wer = wer_metric.compute(
            predictions=self.val_predicted_texts, references=self.val_reference_texts
        )
        # Log the WER
        pl_module.log("val_wer", wer, prog_bar=True, logger=True)
        print("WER on validate data:", wer)
        print("First 5 predictions: ", self.val_predicted_texts[:5])
        print("First 5 references: ", self.val_reference_texts[:5])

        # Clear the lists for the next epoch
        self.val_predicted_texts = []
        self.val_reference_texts = []

    def ctc_decode(self, logits, processor):
        # logits shape: (time, batch, vocab)
        # Transpose to (batch, time, vocab)
        logits = logits.transpose(0, 1)
        # Get the class indices
        class_indices = logits.argmax(dim=2)
        # Remove blanks and collapse repeats for each sequence
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


# # Training

# In[ ]:


# Initialize the data module
datamodule = VietBud500DataModule(batch_size=24, processor_name="vinai/PhoWhisper-tiny")
datamodule.prepare_data()
datamodule.setup()


# Initialize the Lightning module
lightning_module = PhoWhisperLightningModule(
    model_name="vinai/PhoWhisper-tiny", learning_rate=1e-4, warmup_steps=20
)

# print("Evaluate before training", wer_ctc_evaluate(model, datamodule.test_dataloader()))


# Initialize the custom callback
lr_monitor = LearningRateMonitor(logging_interval="step")
checkpoint_callback = ModelCheckpoint(
    monitor="val_wer", mode="min", save_top_k=1, filename="best-{val_wer:.4f}"
)
eval_callback = EvalCallback(processor=datamodule.processor)

# Initialize the PyTorch Lightning Trainer
trainer = pl.Trainer(
    # max_steps=20000,
    max_epochs=64,
    accelerator="gpu",
    devices=1,
    logger=True,
    precision="bf16-mixed",
    callbacks=[lr_monitor, checkpoint_callback, eval_callback],
    # accumulate_grad_batches=1,
)

# Train the model on the training set
trainer.fit(lightning_module, datamodule=datamodule)

# Test the model on the test set
trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")


# In[ ]:
