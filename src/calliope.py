import torch
import torch.nn as nn
from tools.dataset import create_dataloader
from tools.config import CALLIOPE_CONFIG_124M
from tools.utils import calc_loss_loader
from modules.polymnia import Polymnia


with open("../text/alice.txt", "r", encoding="utf-8") as f:
    text = f.read()


train_ratio = 0.9

split_index = int(len(text) * train_ratio)
train_data = text[:split_index]
val_data = text[split_index:]

torch.manual_seed(123)

model = Polymnia(CALLIOPE_CONFIG_124M)
model.eval()

train_loader = create_dataloader(train_data, batch_size=2, max_length=CALLIOPE_CONFIG_124M["context_length"], stride=CALLIOPE_CONFIG_124M["context_length"])
val_loader = create_dataloader(val_data, batch_size=2, max_length=CALLIOPE_CONFIG_124M["context_length"], stride=CALLIOPE_CONFIG_124M["context_length"], shuffle=False, drop_last=False)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print(f"Train Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

