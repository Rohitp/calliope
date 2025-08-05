import torch
import torch.nn as nn
import json 
from train.train_utils import finetune_helper
from train.instruction_data import TrainData, collate

with open("./train/data/instructions-finetune.json","r") as f:
    data = json.load(f)



train_index = int(len(data) * 0.85)
test_index = int(len(data) * 0.10)
validate_index = int(len(data) - train_index - test_index)

train_data = data[:train_index]
test_data = data[train_index:train_index + test_index]
validate_data = data[train_index + test_index:]


inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (inputs_1, inputs_2, inputs_3)
result = collate(batch)
print(result)
