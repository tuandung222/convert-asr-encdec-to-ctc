#!/usr/bin/env python

# In[6]:


# %pip install datasets transformers librosa soundfile jiwer evaluate --quiet
# %pip install pytorch_lightning torch bitsandbytes --quiet


# # Define Datamodule, Model

# In[10]:


import random

import bitsandbytes as bnb
import librosa
import pytorch_lightning as pl
import soundfile as sf
import torch
from bitsandbytes.optim import Adam8bit
from datasets import load_dataset
from easydict import EasyDict as edict
from evaluate import load as load_metric
from IPython.display import Audio
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


# In[2]:


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


# In[3]:


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
        logits = outputs.logits.detach().cpu()
        labels = batch["labels"].detach().cpu()
        predicted_texts = self.ctc_decode(logits, self.processor)
        reference_texts = self.processor.batch_decode(labels, skip_special_tokens=True)
        self.val_predicted_texts.extend(predicted_texts)
        self.val_reference_texts.extend(reference_texts)

    def on_validation_epoch_end(self, trainer, pl_module):
        wer_metric = load_metric("wer")
        wer = wer_metric.compute(
            predictions=self.val_predicted_texts, references=self.val_reference_texts
        )
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


# In[20]:


# Initialize the data module
datamodule = VietBud500DataModule(batch_size=24, processor_name="vinai/PhoWhisper-tiny")
datamodule.prepare_data()
datamodule.setup()


# Initialize the Lightning module
lightning_module = PhoWhisperLightningModule(
    model_name="vinai/PhoWhisper-tiny", learning_rate=1e-4, warmup_steps=20
)


# In[7]:


lightning_module = PhoWhisperLightningModule.load_from_checkpoint(
    "./lightning_logs/version_29/checkpoints/best-val_wer=0.3986.ckpt"
)

print("Evaluate after training", wer_ctc_evaluate(lightning_module, datamodule.test_dataloader()))


# - The evaluation result on test set is 0.41 of WER (Word Error Rate).
# - !� is a bug in text label tokenzation during training. The first two special tokens shouldn't be included in the text label tokenization. It 's my mistake.

# # Upload Best Checkpoint to HuggingFace

# In[16]:


# upload checkpoint ./lightning_logs/version_29/checkpoints/best-val_wer=0.3986.ckpt
# to huggingface at repo: tuandunghcmut/PhoWhisper-tiny-CTC
# import huggingface_hub
# import os

# checkpoint_path = "./lightning_logs/version_29/checkpoints/best-val_wer=0.3986.ckpt"
# model_id = "tuandunghcmut/PhoWhisper-tiny-CTC"
# model_hub = huggingface_hub.HfApi()
# model_hub.create_repo(model_id, exist_ok=True)
# model_hub.upload_file(
#     path_or_fileobj=checkpoint_path,
#     path_in_repo="best-val_wer=0.3986.ckpt",
#     repo_id=model_id,
# )


# # Download checkpoint from HuggingFace hub
# - Everyone can download the checkpoint from HuggingFace hub and use it for inference.

# In[26]:


new_ckpt_path = "best-val_wer=0.3986.ckpt"
from huggingface_hub import hf_hub_download

# download the checkpoint from huggingface
hf_hub_download(
    repo_id=model_id,
    filename="best-val_wer=0.3986.ckpt",
    local_dir="./",
)


# # Load checkpoint

# In[27]:


lightning_module = PhoWhisperLightningModule.load_from_checkpoint("best-val_wer=0.3986.ckpt")


# In[48]:


def predict(model, processor, audio_path, device="cuda"):
    model.to(device)
    audio, _ = librosa.load(audio_path, sr=16000)
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features
    input_features = input_features.to(model.device)
    logits = model(input_features=input_features).logits
    predicted_text = model.ctc_decode(logits, processor)[0]
    predicted_text = predicted_text[2:]  # remove the first token, due to my tokenization mistake
    return predicted_text


def get_random_test_audio(datamodule):
    random_index = random.randint(0, len(datamodule.test_dataset))
    audio_obj = datamodule.test_dataset[random_index]["audio"]
    transcription = datamodule.test_dataset[random_index]["transcription"]
    print("Ground True Transcription:", transcription)
    audio_path = "random_test_audio.wav"
    sf.write(audio_path, audio_obj["array"], 16000)
    display(Audio(audio_path))
    return audio_obj, transcription


def get_random_gd_pred_pairs(datamodule, model, num_examples=10):
    random_indices = random.sample(range(len(datamodule.test_dataset)), num_examples)
    # Print the ground truth and predicted transcriptions, also display the audio
    for index in random_indices:
        audio_obj = datamodule.test_dataset[index]["audio"]
        transcription = datamodule.test_dataset[index]["transcription"]
        audio_path = "random_test_audio.wav"
        sf.write(audio_path, audio_obj["array"], 16000)
        display(Audio(audio_path))
        predicted_text = predict(model, datamodule.processor, audio_path)
        print("Ground True Transcription:", transcription)
        print("Predicted Transcription:", predicted_text)
        print("\n")


get_random_gd_pred_pairs(datamodule, lightning_module, num_examples=25)


# # Evaluate on test set

# - WER result: 0.41

# In[51]:


lightning_module = PhoWhisperLightningModule.load_from_checkpoint("./best-val_wer=0.3986.ckpt")

print("Evaluate after training", wer_ctc_evaluate(lightning_module, datamodule.test_dataloader()))


# - Speed: 1000 examples in 20 seconds <=> each example averagely takes 0.02 seconds
#

#
