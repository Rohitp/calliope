import torch
import torch.nn as nn
from tools.dataset import create_dataloader
from tools.config import CALLIOPE_CONFIG_124M


with open("../text/alice.txt", "r", encoding="utf-8") as f:
    text = f.read()


train_ratio = 0.9

split_index = int(len(text) * train_ratio)
train_data = text[:split_index]
val_data = text[split_index:]

torch.manual_seed(123)

train_loader = create_dataloader(train_data, batch_size=2, max_length=CALLIOPE_CONFIG_124M["context_length"], stride=CALLIOPE_CONFIG_124M["context_length"])
val_loader = create_dataloader(val_data, batch_size=2, max_length=CALLIOPE_CONFIG_124M["context_length"], stride=CALLIOPE_CONFIG_124M["context_length"], shuffle=False, drop_last=False)    



print("Training Data Loader:")
for x, y in train_loader:
    print(f"Input batch shape: {x.shape}, Target batch shape: {y.shape}")

print("\nValidation Data Loader:")
for x, y in val_loader:
    print(f"Input batch shape: {x.shape}, Target batch shape: {y.shape}")
