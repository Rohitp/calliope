# Dump file for random snippets I need to test 

import torch 
import tiktoken
from attention_scores import AttentionScores
from config import CALLIOPE_CONFIG_124M
from polymnia import Polymnia




inputs = torch.tensor([
    [0.43, 0.15, 0.89], 
    [0.55, 0.87, 0.66], 
    [0.57, 0.85, 0.64], 
    [0.22, 0.58, 0.33], 
    [0.77, 0.25, 0.10], 
    [0.05, 0.80, 0.55]
])



x = [1,2, 3]

diction = {
    "name": "Alice",
    "age": 30,
}

def add(a, b, c):
    return a + b + c

def print_u(name, age):
    print(f"{name} is {age} years old")

print(add(*x))
print_u(**diction)

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
for i in ["Every effort moves you", "every day holds a"]:
    batch.append( torch.tensor(tokenizer.encode(i)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = Polymnia(CALLIOPE_CONFIG_124M)
logits = model(batch)
# print(logits.shape)
# print(logits)   

