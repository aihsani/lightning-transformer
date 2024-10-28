import torch
import os
import urllib

from arch.gpt2 import GPTModel
from data import InMemoryDataset
from modules.trainer import TrainingModule
from torch.utils.data import DataLoader


import tiktoken
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L


def main():

    # download training data
    file_path = "data/the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    text_data = ''
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_text = text_data[:split_idx]
    val_text = text_data[split_idx:]

    # configure for training
    GPT_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads":  12,           # Number of attention heads
        "n_layers":  12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    TRAINING_CONFIG = {
        "learning_rate": 5e-4,
        "weight_decay":  1,
        "batch_size":    2,
        "num_epochs":    25,
    }

    DATA_LOADER_CONFIG = {
        'num_workers': 0,
        'drop_last_train': True,
        'shuffle_train': True,
        'drop_last_val': False,
        "shuffle_val": False,
    }

    torch.manual_seed(123)
    llm_model = GPTModel(GPT_CONFIG)
    tokenizer = tiktoken.get_encoding("gpt2")

    training_module = TrainingModule(
        model=llm_model,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        tokenizer=tokenizer,
        test_string="Every effort moves you",
        temperature=0.1,
    )

    training_loader = DataLoader(
        InMemoryDataset(
            txt=train_text,
            tokenizer=tokenizer,
            max_length=GPT_CONFIG["context_length"],
            stride=GPT_CONFIG["context_length"],
        ),
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=DATA_LOADER_CONFIG["shuffle_train"],
        drop_last=DATA_LOADER_CONFIG["drop_last_train"],
        num_workers=DATA_LOADER_CONFIG["num_workers"],
    )

    val_loader = DataLoader(
        InMemoryDataset(
            txt=val_text,
            tokenizer=tokenizer,
            max_length=GPT_CONFIG["context_length"],
            stride=GPT_CONFIG["context_length"],
        ),
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=DATA_LOADER_CONFIG["shuffle_val"],
        drop_last=DATA_LOADER_CONFIG["drop_last_val"],
        num_workers=DATA_LOADER_CONFIG["num_workers"],
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
    )
    trainer = L.Trainer(
        max_epochs=TRAINING_CONFIG["num_epochs"],
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=[1],
        num_sanity_val_steps=TRAINING_CONFIG["batch_size"],
        val_check_interval=1,
        enable_progress_bar=True,
    )

    # Train the model
    trainer.fit(
        model=training_module,
        train_dataloaders=training_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()
