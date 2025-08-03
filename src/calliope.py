import torch
import torch.nn as nn
from tools.dataset import create_dataloader
from tools.config import CALLIOPE_CONFIG_124M, model_configs
from tools.utils import calc_loss_loader, generate_text, text_to_token_ids, token_ids_to_text
from modules.polymnia import Polymnia
from train.train_utils import train_model
from plot.loss_epoch import plot_losses
from tools.gpt_get_weights import download_and_load_gpt, load_gpt2_params, load_settings_and_params
from tools.load_weights import load_weights_into_gpt
import tiktoken



with open("../text/alice.txt", "r", encoding="utf-8") as f:
    text = f.read()


train_ratio = 0.9

split_index = int(len(text) * train_ratio)
train_data = text[:split_index]
val_data = text[split_index:]

torch.manual_seed(123)

model_name = "gpt2-small-124M" # Change this to the desired model size
MODEL_CONFIG = CALLIOPE_CONFIG_124M.copy()
MODEL_CONFIG.update(model_configs[model_name])

model = Polymnia(MODEL_CONFIG)
model.eval()

train_loader = create_dataloader(train_data, batch_size=2, max_length=MODEL_CONFIG["context_length"], stride=MODEL_CONFIG["context_length"])
val_loader = create_dataloader(val_data, batch_size=2, max_length=MODEL_CONFIG["context_length"], stride=MODEL_CONFIG["context_length"], shuffle=False, drop_last=False)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The learning rate is how big a step the optimiser takes after modifying the weights each batch 
# The weight decay is a factor that pulls the learning rate back towards zero, preventing overfitting
# Understaning optimisers and how they work and differences between them was beyond my scope here.
# Using AdamW as it's the most popular. It differs from Adam in that it decouples weight decay from the learning rate, which is supposed to improve performance.

# Further reading on learning rates
# Further reading for more topics -> https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
# And this -> https://spotintelligence.com/2024/04/29/cosine-annealing-in-machine-learning/
# And this to work with learning rates -> https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem

optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_CONFIG["learning_rate"], weight_decay=MODEL_CONFIG["weight_decay"])
tokenizer = tiktoken.get_encoding("gpt2")


weights_path = './weights/124M'

# Load params and test
settings, params = load_settings_and_params(weights_path)


load_weights_into_gpt(model, params)
model.to(device)

token_ids = generate_text(model, text_to_token_ids("The second ammendment", tokenizer).to(device), 25, MODEL_CONFIG["context_length"], temperature=1.5, top_k=50, eos_id=None)
print(token_ids_to_text(token_ids, tokenizer))


# download_and_load_gpt(model, params)

# for name, param in model.named_parameters():
#     print(f"{name}: {param.shape} ({param.numel():,} params)")







