import torch
import torch.nn as nn
from tools.dataset import create_dataloader
from tools.config import CALLIOPE_CONFIG_124M
from tools.utils import calc_loss_loader
from modules.polymnia import Polymnia
from train.train_utils import train_model
from plot.loss_epoch import plot_losses
import tiktoken


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

# The learning rate is how big a step the optimiser takes after modifying the weights each batch 
# The weight decay is a factor that pulls the learning rate back towards zero, preventing overfitting
# Understaning optimisers and how they work and differences between them was beyond my scope here.
# Using AdamW as it's the most popular. It differs from Adam in that it decouples weight decay from the learning rate, which is supposed to improve performance.

# Further reading on learning rates
# Further reading for more topics -> https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
# And this -> https://spotintelligence.com/2024/04/29/cosine-annealing-in-machine-learning/
# And this to work with learning rates -> https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem

optimizer = torch.optim.AdamW(model.parameters(), lr=CALLIOPE_CONFIG_124M["learning_rate"], weight_decay=CALLIOPE_CONFIG_124M["weight_decay"])
tokenizer = tiktoken.get_encoding("gpt2")



train_losses, val_losses, tokens_seen = train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10, eval_freq=5, eval_iter=5, start_context="Hello, I am", tokenizer=tokenizer)


epochs_tensor = torch.linspace(0, 10, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


