import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from malicious_code_test.modeling import MaliciousCodeTest, MaliciousCodeTestConfig


PROJECT_ROOT = Path(__file__).parents[1]
MODEL_DIR = str(PROJECT_ROOT / "model")
DATASET_PATH = str(PROJECT_ROOT / "dataset" / "dataset.csv")

class CustomDataset(torch.utils.data.Dataset):
    """Dataset class holding pairs of input and label sequences."""

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])


if __name__ == "__main__":
    # Set working directory to the script's location
    current_dir = Path(__file__).parent
    os.chdir(current_dir)

    # Configuration
    config = MaliciousCodeTestConfig()
    # config.save_pretrained(save_dir)  # Save config.json

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset from CSV file (expects a 'text' column)
    dataset = load_dataset("csv", data_files=DATASET_PATH)

    # Japanese GPT-2 tokenizer setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress parallelism warnings
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Check if the config vocab size matches the tokenizer's vocab size
    if config.vocab_size != len(tokenizer):
        raise ValueError(
            f"Vocab size mismatch: config.vocab_size={config.vocab_size}, tokenizer.vocab_size={len(tokenizer)}"
        )

    # Convert text to token ID sequences
    def tokenize_function(example):
        ids = tokenizer.encode(example["text"])
        return {"ids": ids}

    # Tokenize the entire dataset
    tokenized = dataset["train"].map(tokenize_function)
    all_ids = sum(tokenized["ids"], [])  # Concatenate all token IDs

    # Create input and label sequences in blocks of block_size
    inputs = []
    labels = []
    block_size = config.block_size
    for i in range(0, len(all_ids) - block_size, block_size):
        x = all_ids[i : i + block_size]
        y = all_ids[i + 1 : i + block_size + 1]
        if len(x) == block_size and len(y) == block_size:
            inputs.append(x)
            labels.append(y)

    # Create PyTorch Dataset and DataLoader
    ds = CustomDataset(inputs, labels)
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=8)

    # Prepare model, optimizer, and loss function
    model = MaliciousCodeTest(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(20):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}, loss: {total_loss/len(dataloader):.4f}")

    # Save the trained model
    model.save_pretrained(MODEL_DIR)  # Save config.json and model weights
    print("Model saved")

    # Load the saved model and run inference
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.to(device)

    # Prepare input
    prompt = "お気に入りの音楽を"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate tokens
    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True
    )

    # Decode and print the result
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(generated_text)
