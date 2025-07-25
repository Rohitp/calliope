import torch
import torch.nn as nn
from tools.dataset import CalliopeDataset


with open("../text/alice.txt", "r", encoding="utf-8") as f:
    text = f.read()


train_ratio = 0.9

split_index = int(len(text) * train_ratio)
train_data = text[:split_index]
val_data = text[split_index:]

torch.manual_seed(123)

train_dataset = CalliopeDataset(train_data)


