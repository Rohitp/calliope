import torch
import torch.nn as nn
import json 
from train.train_utils import finetune_helper, train_model
from train.instruction_data import TrainData, collate, collate_v2
from torch.utils.data import DataLoader
from functools import partial
import tiktoken
from tools.gpt_get_weights import download_and_load_gpt, load_gpt2_params, load_settings_and_params
from modules.polymnia import Polymnia
from tools.config import CALLIOPE_CONFIG_124M, model_configs
from tools.load_weights import load_weights_into_gpt
from tools.utils import calc_loss_loader, generate_text, text_to_token_ids, token_ids_to_text
import time


# For local vs remote training
LOCAL_INSTRUCTIONS = "./train/data/instructions-finetune-local.json"
REMOTE_INSTRUCTIONS = "./train/data/instructions-finetune.json"

with open(REMOTE_INSTRUCTIONS,"r") as f:
    data = json.load(f)


NUM_WORKERS = 0
BATCH_SIZE = 8
WEIGHTS_PATH = './weights/'
MODEL_SIZE = "774M"
NUM_EPOCHS = 5
EVAL_FREQ = 5
EVAL_ITER = 5


model_name = "gpt2-large-774M" # Change this to the desired model size
MODEL_CONFIG = CALLIOPE_CONFIG_124M.copy()
MODEL_CONFIG.update(model_configs[model_name])

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")


train_index = int(len(data) * 0.85)
test_index = int(len(data) * 0.10)
validate_index = int(len(data) - train_index - test_index)

train_data = data[:train_index]
test_data = data[train_index:train_index + test_index]
validate_data = data[train_index + test_index:]



# device can also be "mps" for Apple Silicon Macs
# MPS -> Metal Performance Shaders, which is a framework for GPU-accelerated computing on Apple devices
# https://docs.pytorch.org/docs/stable/notes/mps.html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():
# device = torch.device("mps")"


collate_v3 = partial(collate_v2, device=device, max_length=1024)   


train_dataset = TrainData(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_v3, num_workers=NUM_WORKERS)


val_dataset = TrainData(validate_data, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_v3, num_workers=NUM_WORKERS)

test_dataset = TrainData(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_v3, num_workers=NUM_WORKERS)

# settings, params = download_and_load_gpt("774M", WEIGHTS_PATH)
settings, params = load_settings_and_params(WEIGHTS_PATH + MODEL_SIZE)


model = Polymnia(MODEL_CONFIG)
load_weights_into_gpt(model, params)
# model.eval()


# input_data = finetune_helper(validate_data[0])


# token_ids = generate_text(model, text_to_token_ids(input_data, tokenizer), 50, MODEL_CONFIG["context_length"], eos_id=50256)
# generate_text = token_ids_to_text(token_ids, tokenizer)

# print(f"Generated text: {generate_text}")


model.to(device)
# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
#     val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)


optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_CONFIG["learning_rate"], weight_decay=MODEL_CONFIG["weight_decay"])

start_time = time.time()
start_context = finetune_helper(validate_data[0])

train_losses, val_losses, track_tokens_seen = train_model(model, train_loader, val_loader, optimizer, device, NUM_EPOCHS, EVAL_FREQ, EVAL_ITER, start_context, tokenizer)

end_time = time.time()
execution_time = (end_time - start_time) / 60


torch.save(model.state_dict(),"weights/model-774m-al00-v1.pth")

